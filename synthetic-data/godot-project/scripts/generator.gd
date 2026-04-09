class_name SyntheticGenerator
extends Node

# Generates synthetic egocentric grasping data
# Outputs: video frames + physics metadata (position, velocity, acceleration)

signal generation_complete(batch_id: String, metadata_path: String)
signal frame_captured(frame_number: int, total_frames: int)

@export var output_dir: String = "res://output/"
@export var batch_size: int = 100  # Number of variations to generate
@export var capture_duration: float = 3.0  # seconds per variation
@export var fps: int = 30

var current_batch: int = 0
var current_frame: int = 0
var is_generating: bool = false
var metadata: Dictionary = {}

# Scene references
@onready var scene_root: Node3D
@onready var camera: Camera3D
@onready var object: RigidBody3D
@onready var grabber: Node3D

func _ready():
	# Ensure output directory exists
	var dir = DirAccess.open("user://")
	if dir:
		dir.make_dir_recursive("physicalfish_output")

func setup_scene_refs(root: Node3D, cam: Camera3D, obj: RigidBody3D, grab: Node3D):
	scene_root = root
	camera = cam
	object = obj
	grabber = grab

func generate_batch(batch_id: String = "") -> Dictionary:
	if batch_id.is_empty():
		batch_id = "batch_%s" % Time.get_datetime_string_from_system().replace(":", "-")
	
	is_generating = true
	current_batch = 0
	
	var batch_metadata = {
		"batch_id": batch_id,
		"timestamp": Time.get_unix_time_from_system(),
		"variations": [],
		"global_params": Config.get_all_params()
	}
	
	print("Starting synthetic batch generation: %s" % batch_id)
	
	for i in range(batch_size):
		current_batch = i
		var variation_id = "%s_var_%04d" % [batch_id, i]
		
		# Randomize physics parameters for this variation
		Config.randomize_params()
		
		# Reset scene
		_reset_scene()
		
		# Generate variation
		var variation_data = await _generate_variation(variation_id)
		batch_metadata["variations"].append(variation_data)
		
		print("Generated variation %d/%d: %s" % [i + 1, batch_size, variation_id])
	
	# Save metadata
	var metadata_path = _save_metadata(batch_metadata)
	is_generating = false
	
	generation_complete.emit(batch_id, metadata_path)
	
	return batch_metadata

func _reset_scene():
	# Reset object position
	if object:
		object.position = Vector3(0, 1.0, -0.5)
		object.linear_velocity = Vector3.ZERO
		object.angular_velocity = Vector3.ZERO
		
		# Apply current physics parameters
		object.mass = Config.object_mass
		object.physics_material_override = PhysicsMaterial.new()
		object.physics_material_override.friction = Config.object_friction
		object.physics_material_override.bounce = Config.object_bounce
		object.linear_damp = Config.linear_damping
		object.angular_damp = Config.angular_damping
	
	# Reset grabber
	if grabber:
		grabber.position = Vector3(0, 0.5, -0.3)

func _generate_variation(variation_id: String) -> Dictionary:
	var variation_data = {
		"variation_id": variation_id,
		"physics_params": Config.get_all_params(),
		"frames": [],
		"events": []
	}
	
	var total_frames = int(capture_duration * fps)
	var grab_frame = int(total_frames * 0.3)  # Grab at 30% of sequence
	var release_frame = int(total_frames * 0.7)  # Release at 70%
	
	# Capture frames
	for frame in range(total_frames):
		current_frame = frame
		
		# Simulate grabber action
		if frame == grab_frame:
			_execute_grab()
			variation_data["events"].append({"frame": frame, "event": "grab"})
		elif frame == release_frame:
			_execute_release()
			variation_data["events"].append({"frame": frame, "event": "release"})
		
		# Step physics
		await get_tree().physics_frame
		
		# Capture frame data
		var frame_data = _capture_frame_data(frame)
		variation_data["frames"].append(frame_data)
		
		frame_captured.emit(frame, total_frames)
	
	return variation_data

func _execute_grab():
	# Simplified grab logic - attach object to grabber
	if object and grabber:
		# In a real implementation, this would use a joint or parenting
		# For demo purposes, we apply an upward force
		object.apply_central_impulse(Vector3(0, Config.grabber_force * 0.1, 0))

func _execute_release():
	# Release logic - let object fall
	if object:
		# Apply slight random velocity for variation
		var random_vel = Vector3(
			randf_range(-0.5, 0.5),
			randf_range(0, 0.3),
			randf_range(-0.5, 0.5)
		)
		object.apply_central_impulse(random_vel)

func _capture_frame_data(frame_number: int) -> Dictionary:
	var timestamp = frame_number / float(fps)
	
	var frame_data = {
		"frame": frame_number,
		"timestamp": timestamp,
		"object": {},
		"camera": {},
	}
	
	if object:
		frame_data["object"] = {
			"position": [object.position.x, object.position.y, object.position.z],
			"velocity": [object.linear_velocity.x, object.linear_velocity.y, object.linear_velocity.z],
			"acceleration": _calculate_acceleration(object),
			"rotation": [object.rotation.x, object.rotation.y, object.rotation.z],
			"angular_velocity": [object.angular_velocity.x, object.angular_velocity.y, object.angular_velocity.z],
		}
	
	if camera:
		frame_data["camera"] = {
			"position": [camera.position.x, camera.position.y, camera.position.z],
			"rotation": [camera.rotation.x, camera.rotation.y, camera.rotation.z],
			"fov": camera.fov,
		}
	
	return frame_data

func _calculate_acceleration(body: RigidBody3D) -> Array:
	# Calculate acceleration from velocity change (simplified)
	# In a real implementation, you'd track previous velocity
	var gravity = Vector3(0, -Config.gravity_magnitude, 0)
	var accel = gravity  # Start with gravity
	
	# Add any applied forces (simplified)
	if body.get_colliding_bodies().size() > 0:
		accel += Vector3(0, Config.gravity_magnitude * 0.5, 0)  # Approximate contact force
	
	return [accel.x, accel.y, accel.z]

func _save_metadata(batch_metadata: Dictionary) -> String:
	var filename = "%s_metadata.json" % batch_metadata["batch_id"]
	var path = "user://physicalfish_output/%s" % filename
	
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(batch_metadata, "\t"))
		file.close()
		print("Saved metadata to: %s" % path)
		return path
	else:
		push_error("Failed to save metadata")
		return ""

# Headless generation entry point
func run_headless_generation(batch_id: String = ""):
	print("PhysicalFish Synthetic Generator — Headless Mode")
	print("Batch ID: %s" % (batch_id if not batch_id.is_empty() else "auto"))
	
	# Wait for scene to be ready
	await get_tree().create_timer(1.0).timeout
	
	var metadata = await generate_batch(batch_id)
	
	print("Generation complete!")
	print("Variations generated: %d" % metadata["variations"].size())
	
	# Exit after generation (for headless mode)
	get_tree().quit()
