#pragma once

// Небольшая библиотека векторов и матриц для лабораторной.
// Матрицы хранятся по столбцам: m[столбец][строка], так же как mat4 в GLSL,
// поэтому их можно сразу копировать в uniform-буфер.
// Мировая система координат правая, ось Y смотрит вверх, камера смотрит вдоль -Z.
// Матрицы проекции сделаны под Vulkan: ось Y на экране направлена вниз, глубина от 0 до 1.

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

// Перевод градусов в радианы
inline float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

struct Vec3 {
	float x, y, z;
};

struct Vec4 {
	float x, y, z, w;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator-(Vec3 a) { return { -a.x, -a.y, -a.z }; }
inline Vec3 operator*(Vec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }

// Скалярное произведение
inline float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Векторное произведение
inline Vec3 cross(Vec3 a, Vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline float length(Vec3 v) {
	return std::sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 v) {
	return v * (1.0f / length(v));
}

struct Mat4 {
	float m[4][4]; // m[столбец][строка]
};

// Единичная матрица
inline Mat4 identity() {
	Mat4 r = {};
	r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
	return r;
}

// Умножение матриц: элемент (i, j) равен сумме a(i, k) * b(k, j)
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r = {};
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			for (int k = 0; k < 4; ++k) {
				r.m[j][i] += a.m[k][i] * b.m[j][k];
			}
		}
	}
	return r;
}

// Перенос: вектор переноса записывается в последний столбец
inline Mat4 translate(Vec3 t) {
	Mat4 r = identity();
	r.m[3][0] = t.x;
	r.m[3][1] = t.y;
	r.m[3][2] = t.z;
	return r;
}

// Растяжение по осям
inline Mat4 scale(Vec3 s) {
	Mat4 r = {};
	r.m[0][0] = s.x;
	r.m[1][1] = s.y;
	r.m[2][2] = s.z;
	r.m[3][3] = 1.0f;
	return r;
}

// Поворот вокруг оси X на угол в радианах
inline Mat4 rotateX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[1][1] = c;  r.m[2][1] = -s;
	r.m[1][2] = s;  r.m[2][2] = c;
	return r;
}

// Поворот вокруг оси Y
inline Mat4 rotateY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0][0] = c;   r.m[2][0] = s;
	r.m[0][2] = -s;  r.m[2][2] = c;
	return r;
}

// Поворот вокруг оси Z
inline Mat4 rotateZ(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.m[0][0] = c;  r.m[1][0] = -s;
	r.m[0][1] = s;  r.m[1][1] = c;
	return r;
}

// Поворот углами Эйлера: сначала вокруг X, потом Y, потом Z
inline Mat4 rotateEuler(Vec3 angles) {
	return rotateZ(angles.z) * rotateY(angles.y) * rotateX(angles.x);
}

// Матрица вида: камера в точке eye смотрит на точку target.
// Строки матрицы это оси камеры: вправо, вверх и назад.
inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
	const Vec3 f = normalize(target - eye);
	const Vec3 s = normalize(cross(f, up));
	const Vec3 u = cross(s, f);

	Mat4 r = identity();
	r.m[0][0] = s.x;  r.m[1][0] = s.y;  r.m[2][0] = s.z;
	r.m[0][1] = u.x;  r.m[1][1] = u.y;  r.m[2][1] = u.z;
	r.m[0][2] = -f.x; r.m[1][2] = -f.y; r.m[2][2] = -f.z;
	r.m[3][0] = -dot(s, eye);
	r.m[3][1] = -dot(u, eye);
	r.m[3][2] = dot(f, eye);
	return r;
}

// Перспективная проекция для Vulkan.
// Минус у элемента [1][1] переворачивает ось Y, потому что в Vulkan она смотрит вниз.
// Точка на ближней плоскости получает глубину 0, на дальней глубину 1.
// В w записывается -z, после деления на w далёкие объекты становятся меньше.
inline Mat4 perspective(float fov_y, float aspect, float near_plane, float far_plane) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);

	Mat4 r = {};
	r.m[0][0] = f / aspect;
	r.m[1][1] = -f;
	r.m[2][2] = far_plane / (near_plane - far_plane);
	r.m[2][3] = -1.0f;
	r.m[3][2] = near_plane * far_plane / (near_plane - far_plane);
	return r;
}

// Ортографическая проекция для Vulkan.
// Переводит параллелепипед видимости в куб от -1 до 1 по x и y и от 0 до 1 по глубине.
// w остаётся равным 1, поэтому размер объектов не зависит от расстояния.
inline Mat4 orthographic(float left, float right, float bottom, float top,
                         float near_plane, float far_plane) {
	Mat4 r = {};
	r.m[0][0] = 2.0f / (right - left);
	r.m[1][1] = -2.0f / (top - bottom);
	r.m[2][2] = -1.0f / (far_plane - near_plane);
	r.m[3][0] = -(right + left) / (right - left);
	r.m[3][1] = (top + bottom) / (top - bottom);
	r.m[3][2] = -near_plane / (far_plane - near_plane);
	r.m[3][3] = 1.0f;
	return r;
}

} // namespace math
