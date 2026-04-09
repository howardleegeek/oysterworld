class_name SceneController
extends Node3D

# Main scene controller that wires up the generator with scene references

@onready var object: RigidBody3D = $ObjectToGrab
@onready var grabber: Node3D = $Grabber
@onready var camera: Camera3D = $CameraRig/EgocentricCamera
@onready var status_label: Label = $UI/StatusLabel
@onready var progress_label: Label = $UI/ProgressLabel

var is_headless: bool = false
var target_batch_id: String = ""

func _ready():
	print("PhysicalFish Scene Controller initialized")
	
	# Check for headless mode
	var args = OS.get_cmdline_args()
	for arg in args:
		if arg.begins_with("--batch-id="):
			target_batch_id = arg.replace("--batch-id=", "")
			is_headless = true
		elif arg == "--headless":
			is_headless = true
	
	# Setup generator with scene references
	Generator.setup_scene_refs(self, camera, object, grabber)
	
	# Connect signals
	Generator.generation_complete.connect(_on_generation_complete)
	Generator.frame_captured.connect(_on_frame_captured)
	
	# Start generation
	if is_headless:
		print("Running in headless mode")
		Generator.run_headless_generation(target_batch_id)
	else:
		# Interactive mode - start after brief delay
		await get_tree().create_timer(0.5).timeout
		_start_interactive_generation()

func _start_interactive_generation():
	status_label.text = "Generating synthetic data..."
	
	var batch_id = "demo_%s" % Time.get_datetime_string_from_system().replace(":", "-")
	var metadata = await Generator.generate_batch(batch_id)
	
	status_label.text = "Generation complete! Saved to output/"

func _on_generation_complete(batch_id: String, metadata_path: String):
	print("Batch %s complete! Metadata: %s" % [batch_id, metadata_path])
	
	if not is_headless:
		status_label.text = "Complete: %s" % batch_id

func _on_frame_captured(frame_number: int, total_frames: int):
	if not is_headless:
		progress_label.text = "Progress: %d/%d" % [frame_number + 1, total_frames]

func _input(event):
	if event.is_action_pressed("ui_cancel"):
		get_tree().quit()
	elif event.is_action_pressed("ui_accept"):
		# Manual trigger for testing
		_start_interactive_generation()
