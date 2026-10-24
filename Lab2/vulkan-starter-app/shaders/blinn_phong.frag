#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;

// Такие же блоки, как в вершинном шейдере
layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_projection;
	vec3 camera_position;
	float padding0;
	vec3 ambient_color;
	float ambient_intensity;
	vec3 light_direction; // куда светит направленный источник
	float light_intensity;
	vec3 light_color;
	float padding1;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
	mat4 transform;
	mat4 normal_matrix;
	vec3 albedo; // цвет рассеянного отражения
	float shininess; // коэффициент блика
	vec3 specular_color; // цвет блика
	float padding;
} model;

// Совпадает со struct PointLightData
struct PointLight {
	vec3 position;
	float intensity;
	vec3 color;
	float padding;
};

// Совпадает со struct SpotLightData
struct SpotLight {
	vec3 position;
	float intensity;
	vec3 direction; // куда светит прожектор, единичный вектор
	float cos_inner; // косинус внутреннего угла
	vec3 color;
	float cos_outer; // косинус внешнего угла
};

// Storage-буферы с источниками. В начале буфера лежит количество источников, за ним массив.
// Размер массива в шейдере не указан, его длину задаёт буфер на стороне программы.
layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
	uint count;
	PointLight lights[];
} point_lights;

layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
	uint count;
	SpotLight lights[];
} spot_lights;

layout(location = 0) out vec4 out_color;

// Модель Блинна-Фонга для одного источника.
// n это нормаль, l направление на источник, v направление на камеру, все векторы единичные.
// radiance это цвет источника, умноженный на интенсивность и затухание.
vec3 blinnPhong(vec3 n, vec3 l, vec3 v, vec3 radiance) {
	float n_dot_l = max(dot(n, l), 0.0);
	vec3 diffuse = model.albedo * n_dot_l;

	// Вектор посередине между направлением на источник и направлением на камеру
	vec3 h = normalize(l + v);
	// Блик бывает только на стороне, которая повёрнута к источнику
	float specular_factor = n_dot_l > 0.0 ? pow(max(dot(n, h), 0.0), model.shininess) : 0.0;
	vec3 specular = model.specular_color * specular_factor;

	return (diffuse + specular) * radiance;
}

// Затухание по закону обратных квадратов: освещённость падает как 1 / d^2.
// Квадрат расстояния ограничен снизу, чтобы не делить на ноль.
float inverseSquare(float dist) {
	return 1.0 / max(dist * dist, 0.0001);
}

void main() {
	// После интерполяции нормаль может стать не единичной
	vec3 n = normalize(in_world_normal);
	vec3 v = normalize(scene.camera_position - in_world_position);

	// Фоновый свет одинаково освещает все грани
	vec3 color = scene.ambient_color * scene.ambient_intensity * model.albedo;

	// Направленный источник: лучи идут вдоль light_direction, значит направление на источник обратное.
	// Затухания нет, источник считается бесконечно далёким.
	vec3 sun_l = normalize(-scene.light_direction);
	color += blinnPhong(n, sun_l, v, scene.light_color * scene.light_intensity);

	// Точечные источники
	for (uint i = 0u; i < point_lights.count; ++i) {
		PointLight light = point_lights.lights[i];
		vec3 to_light = light.position - in_world_position;
		float dist = max(length(to_light), 0.0001);
		vec3 l = to_light / dist;

		vec3 radiance = light.color * light.intensity * inverseSquare(dist);
		color += blinnPhong(n, l, v, radiance);
	}

	// Прожекторы
	for (uint i = 0u; i < spot_lights.count; ++i) {
		SpotLight light = spot_lights.lights[i];
		vec3 to_light = light.position - in_world_position;
		float dist = max(length(to_light), 0.0001);
		vec3 l = to_light / dist;

		// Косинус угла между осью прожектора и направлением от прожектора на точку
		float cos_theta = dot(-l, light.direction);

		// Мягкий край: внутри внутреннего угла множитель 1, за внешним 0,
		// между ними линейно убывает по косинусу угла.
		// Если внешний угол не больше внутреннего, край получается резким.
		float edge = light.cos_inner - light.cos_outer;
		float cone = edge > 0.0
			? clamp((cos_theta - light.cos_outer) / edge, 0.0, 1.0)
			: step(light.cos_outer, cos_theta);

		vec3 radiance = light.color * light.intensity * inverseSquare(dist) * cone;
		color += blinnPhong(n, l, v, radiance);
	}

	out_color = vec4(color, 1.0);
}
