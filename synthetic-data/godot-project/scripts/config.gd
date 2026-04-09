class_name PhysicalConfig
extends Node

# Physics parameters that can be optimized by AutoResearch loop
# These are exposed for external control via JSON-RPC

@export_group("Gravity")
@export var gravity_magnitude: float = 9.8  # m/s²
@export var gravity_direction: Vector3 = Vector3(0, -1, 0)

@export_group("Object Properties")
@export var object_mass: float = 0.5  # kg (0.1 - 2.0)
@export var object_friction: float = 0.5  # (0.1 - 1.0)
@export var object_bounce: float = 0.3  # (0.0 - 1.0)

@export_group("Hand/Grabber Properties")
@export var grabber_mass: float = 0.2  # kg
@export var grabber_force: float = 10.0  # N
@export var grabber_speed: float = 1.0  # m/s

@export_group("Damping")
@export var linear_damping: float = 0.1  # (0.0 - 1.0)
@export var angular_damping: float = 0.1  # (0.0 - 1.0)

@export_group("Camera")
@export var camera_fov: float = 70.0  # degrees
@export var camera_height: float = 1.6  # meters (egocentric height)
@export var camera_shake: float = 0.0  # (0.0 - 0.1)

# Parameter bounds for optimization
const BOUNDS = {
	"gravity_magnitude": {"min": 8.0, "max": 11.0},
	"object_mass": {"min": 0.1, "max": 2.0},
	"object_friction": {"min": 0.1, "max": 1.0},
	"object_bounce": {"min": 0.0, "max": 1.0},
	"linear_damping": {"min": 0.0, "max": 1.0},
	"angular_damping": {"min": 0.0, "max": 1.0},
}

func _ready():
	apply_to_physics_server()

func apply_to_physics_server():
	# Apply gravity
	PhysicsServer3D.area_set_param(
		get_viewport().find_world_3d().space,
		PhysicsServer3D.AREA_PARAM_GRAVITY,
		gravity_magnitude
	)
	PhysicsServer3D.area_set_param(
		get_viewport().find_world_3d().space,
		PhysicsServer3D.AREA_PARAM_GRAVITY_VECTOR,
		gravity_direction
	)

func get_all_params() -> Dictionary:
	return {
		"gravity_magnitude": gravity_magnitude,
		"gravity_direction": [gravity_direction.x, gravity_direction.y, gravity_direction.z],
		"object_mass": object_mass,
		"object_friction": object_friction,
		"object_bounce": object_bounce,
		"grabber_mass": grabber_mass,
		"grabber_force": grabber_force,
		"grabber_speed": grabber_speed,
		"linear_damping": linear_damping,
		"angular_damping": angular_damping,
		"camera_fov": camera_fov,
		"camera_height": camera_height,
		"camera_shake": camera_shake,
	}

func set_params_from_dict(params: Dictionary):
	if params.has("gravity_magnitude"):
		gravity_magnitude = params["gravity_magnitude"]
	if params.has("object_mass"):
		object_mass = params["object_mass"]
	if params.has("object_friction"):
		object_friction = params["object_friction"]
	if params.has("object_bounce"):
		object_bounce = params["object_bounce"]
	if params.has("linear_damping"):
		linear_damping = params["linear_damping"]
	if params.has("angular_damping"):
		angular_damping = params["angular_damping"]
	
	apply_to_physics_server()

func randomize_params():
	# Randomize within bounds for synthetic variation generation
	gravity_magnitude = randf_range(BOUNDS["gravity_magnitude"]["min"], BOUNDS["gravity_magnitude"]["max"])
	object_mass = randf_range(BOUNDS["object_mass"]["min"], BOUNDS["object_mass"]["max"])
	object_friction = randf_range(BOUNDS["object_friction"]["min"], BOUNDS["object_friction"]["max"])
	object_bounce = randf_range(BOUNDS["object_bounce"]["min"], BOUNDS["object_bounce"]["max"])
	linear_damping = randf_range(BOUNDS["linear_damping"]["min"], BOUNDS["linear_damping"]["max"])
	angular_damping = randf_range(BOUNDS["angular_damping"]["min"], BOUNDS["angular_damping"]["max"])
	
	apply_to_physics_server()
