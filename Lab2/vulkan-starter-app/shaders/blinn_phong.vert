#version 450

// Входные данные вершины, порядок как в struct Vertex
layout(location = 0) in vec3 in_position; // позиция в локальных координатах объекта
layout(location = 1) in vec3 in_normal; // нормаль грани в локальных координатах

// Набор 0: данные сцены, общие для всех объектов. Совпадает со struct SceneData.
layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_projection;
	vec3 camera_position;
	float padding0;
	vec3 ambient_color;
	float ambient_intensity;
	vec3 light_direction;
	float light_intensity;
	vec3 light_color;
	float padding1;
} scene;

// Набор 1: данные одного объекта. Совпадает со struct ModelData.
layout(set = 1, binding = 0, std140) uniform ModelData {
	mat4 transform;
	mat4 normal_matrix; // mat4 вместо mat3, потому что mat3 в std140 выравнивается неудобно
	vec3 albedo;
	float shininess;
	vec3 specular_color;
	float padding;
} model;

// Освещение считается во фрагментном шейдере в мировых координатах
layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;

void main() {
	vec4 world_position = model.transform * vec4(in_position, 1.0);
	out_world_position = world_position.xyz;
	out_world_normal = mat3(model.normal_matrix) * in_normal;
	gl_Position = scene.view_projection * world_position;
}
