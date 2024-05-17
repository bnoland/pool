#version 430

uniform float u_time;
uniform vec2 u_resolution;

out vec4 frag_color;

/* ----------------------- Constants / Types / Globals ---------------------- */

#define CAMERA_MOVEMENT
// #define REFLECT_ENTIRE_SCENE

struct DistMat
{
  float dist;
  int mat;
};

struct PointMat
{
  vec3 p;
  int mat;
};

struct Light
{
  vec3 pos;
  vec3 color;
};

struct Material
{
  vec3 color;
  float ambient;
  float diffuse;
  float specular;
  float shininess;
};

const float INFINITY = 1.0 / 0.0;

const int MATERIAL_NONE = 0;
const int MATERIAL_POOL = 1;
const int MATERIAL_GROUND = 2;
const int MATERIAL_WALL_XY = 3;
const int MATERIAL_WALL_ZY = 4;
const int MATERIAL_CEILING = 5;
const int MATERIAL_LIGHT1 = 6;
const int MATERIAL_LIGHT2 = 7;
const int MATERIAL_LIGHT3 = 8;
const int MATERIAL_LIGHT4 = 9;
const int MATERIAL_WATER = 10;
const int MATERIAL_BALL = 11;

const float POOL_SIZE_X = 8.0;
const float POOL_SIZE_Z = 8.0;

int g_active_light = int(0.5 * u_time) % 4;

/* ---------------------------- General Utilities --------------------------- */

DistMat sd_scene(in vec3 p);

DistMat cast_ray(in vec3 ray_origin, in vec3 ray_dir)
{
  const float min_t = 0.0;
  const float max_t = 50.0;
  float t = min_t;

  for (int i = 0; i < 2048 && t <= max_t; i++) {
    vec3 p = ray_origin + t * ray_dir;
    DistMat scene = sd_scene(p);
    if (scene.dist < 0.001) {
      return DistMat(t, scene.mat);
    }
    t += scene.dist;
  }

  return DistMat(t, MATERIAL_NONE);
}

vec3 calc_normal(in vec3 p)
{
  const vec2 eps = vec2(0.001, 0.0);
  // XXX: Try to prevent inlining here?
  vec3 n = vec3(
    sd_scene(p + eps.xyy).dist - sd_scene(p - eps.xyy).dist,
    sd_scene(p + eps.yxy).dist - sd_scene(p - eps.yxy).dist,
    sd_scene(p + eps.yyx).dist - sd_scene(p - eps.yyx).dist
  );
  return normalize(n);
}

DistMat sd_scene_no_water(in vec3 p);

float shadow(in vec3 p, in vec3 light_dir)
{
  const float min_t = 0.1;
  const float max_t = 8.0;
  float t = min_t;
  float result = 1.0;

  for (int i = 0; i < 256 && t <= max_t; i++) {
    DistMat scene = sd_scene_no_water(p + t * light_dir);

    // Ignore shadows in the vicinity of the active light.
    if (g_active_light == 0 && scene.mat == MATERIAL_LIGHT1) return 1.0;
    if (g_active_light == 1 && scene.mat == MATERIAL_LIGHT2) return 1.0;
    if (g_active_light == 2 && scene.mat == MATERIAL_LIGHT3) return 1.0;
    if (g_active_light == 3 && scene.mat == MATERIAL_LIGHT4) return 1.0;

    result = min(result, 16.0 * scene.dist / t);
    if (result < -1.0) break;
    t += clamp(scene.dist, 0.01, 10.0);
  }

  result = max(result, -1.0);
  return 0.25 * (1.0 + result) * (1.0 + result) * (2.0 - result);
}

vec3 calc_color(in vec3 p, in vec3 n, in vec3 camera, in Light light, in Material mat)
{
  vec3 light_dir = normalize(light.pos - p);
  vec3 view_dir = normalize(camera - p);

  float diff_coeff = max(0.0, dot(light_dir, n));
  float spec_coeff = pow(max(0.0, dot(reflect(-light_dir, n), view_dir)), mat.shininess);

  float ambient = mat.ambient;
  float diffuse = mat.diffuse * diff_coeff;
  float specular = mat.specular * spec_coeff;
  float intensity = (ambient + diffuse + specular) * shadow(p, light_dir);

  return intensity * light.color * mat.color;
}

/* ---------------------------------- Noise --------------------------------- */

float hash(in vec3 p)
{
  int n = int(p.x * 3.0 + p.y * 113.0 + p.z * 311.0);
  n = (n << 13) ^ n;
  n = n * (n * n * 15731 + 789221) + 1376312589;
  return -1.0 + 2.0 * float(n & 0x0fffffff) / float(0x0fffffff);
}

float noise(in vec3 p)
{
  vec3 m = floor(p);
  vec3 q = fract(p);

  vec3 u = q * q * (3.0 - 2.0 * q);
  vec3 du = 6.0 * q * (1.0 - q);

  // XXX: Prevent inlining?
  float a = hash(m + vec3(0, 0, 0));
  float b = hash(m + vec3(1, 0, 0));
  float c = hash(m + vec3(0, 1, 0));
  float d = hash(m + vec3(1, 1, 0));
  float e = hash(m + vec3(0, 0, 1));
  float f = hash(m + vec3(1, 0, 1));
  float g = hash(m + vec3(0, 1, 1));
  float h = hash(m + vec3(1, 1, 1));

  float k0 = a;
  float k1 = b - a;
  float k2 = c - a;
  float k3 = e - a;
  float k4 = a - b - c + d;
  float k5 = a - c - e + g;
  float k6 = a - b - e + f;
  float k7 = -a + b + c - d + e - f - g + h;

  float value = -1.0 + 2.0 *
    (k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y +
      k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z);

  return value;
}

vec4 noise_d(in vec3 p)
{
  vec3 m = floor(p);
  vec3 q = fract(p);

  vec3 u = q * q * (3.0 - 2.0 * q);
  vec3 du = 6.0 * q * (1.0 - q);

  // XXX: Prevent inlining?
  float a = hash(m + vec3(0, 0, 0));
  float b = hash(m + vec3(1, 0, 0));
  float c = hash(m + vec3(0, 1, 0));
  float d = hash(m + vec3(1, 1, 0));
  float e = hash(m + vec3(0, 0, 1));
  float f = hash(m + vec3(1, 0, 1));
  float g = hash(m + vec3(0, 1, 1));
  float h = hash(m + vec3(1, 1, 1));

  float k0 = a;
  float k1 = b - a;
  float k2 = c - a;
  float k3 = e - a;
  float k4 = a - b - c + d;
  float k5 = a - c - e + g;
  float k6 = a - b - e + f;
  float k7 = -a + b + c - d + e - f - g + h;

  float value = -1.0 + 2.0 *
    (k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y +
      k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z);

  vec3 grad = 2.0 * du *
    vec3(
      k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
      k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
      k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);

  return vec4(value, grad);
}

/* ---------------------------------- Water --------------------------------- */

float water_height(in vec2 p)
{
  float value = 0.0;
  float freq = 1.0;
  float amp = 1.0;

  for (int i = 0; i < 3; i++) {
    value += amp * noise(vec3(freq * p + 0.8 * u_time, 0.8 * u_time));
    freq *= 2.0;
    amp /= 2.0;
  }

  return value / 20.0;
}

float sd_ball(in vec3 p);

PointMat water_reflection(in vec3 p, in vec3 n, in vec3 camera, in Light light)
{
  const float max_t = 50.0;
  const float min_t = 0.01;
  float t = min_t;
  vec3 q = p;

  vec3 light_dir = normalize(light.pos - p);
  vec3 ray_dir = reflect(-light_dir, n);

  for (int i = 0; i < 256 && t <= max_t; i++) {
    q = 0.002 * n + p + t * ray_dir;
#ifdef REFLECT_ENTIRE_SCENE
    DistMat scene = sd_scene_no_water(q);
    if (scene.dist < 0.001) {
      return PointMat(q, scene.mat);
    }
    t += scene.dist;
#else
    float dist = sd_ball(q);
    if (dist < 0.001) {
      return PointMat(q, MATERIAL_BALL);
    }
    t += dist;
#endif
  }

  return PointMat(q, MATERIAL_WATER);
}

float sd_pool(in vec3 p);

PointMat water_refraction(in vec3 p, in vec3 n, in vec3 camera, in Light light)
{
  const float max_t = 10.0;
  const float min_t = 0.01;
  float t = min_t;
  vec3 q = p;

  vec3 view_dir = normalize(camera - p);
  vec3 ray_dir = refract(-view_dir, n, 1.0 / 1.33);

  for (int i = 0; i < 256 && t <= max_t; i++) {
    q = -0.002 * n + p + t * ray_dir;
    float pool = sd_pool(q);
    float ball = sd_ball(q);
    float dist = min(pool, ball);
    if (dist < 0.001) {
      return PointMat(q, (dist == pool) ? MATERIAL_POOL : MATERIAL_BALL);
    }
    t += dist;
  }

  return PointMat(q, MATERIAL_NONE);
}

/* ---------------------------------- Scene --------------------------------- */

float sd_box(in vec3 p, in vec3 s)
{
  vec3 q = abs(p) - s;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sd_sphere(in vec3 p, float r)
{
  return length(p) - r;
}

float sd_pool(in vec3 p)
{
  float edge1 = sd_box(p - vec3(-POOL_SIZE_X, 1.8, 0.0), vec3(0.3, 1.8, POOL_SIZE_Z + 0.3));
  float edge2 = sd_box(p - vec3(POOL_SIZE_X, 1.8, 0.0), vec3(0.3, 1.8, POOL_SIZE_Z + 0.3));
  float edge3 = sd_box(p - vec3(0.0, 1.8, POOL_SIZE_Z), vec3(POOL_SIZE_X, 1.8, 0.3));
  float edge4 = sd_box(p - vec3(0.0, 1.8, -POOL_SIZE_Z), vec3(POOL_SIZE_X, 1.8, 0.3));
  float bottom = sd_box(p - vec3(0.0, 0.05, 0.0), vec3(POOL_SIZE_X, 0.1, POOL_SIZE_Z));

  float dist = min(edge1, min(edge2, min(edge3, min(edge4, bottom))));
  dist -= 0.1;  // Smooth edges
  dist += 0.003 * noise(30.0 * p);  // Stipple

  return dist;
}

DistMat sd_room(in vec3 p)
{
  float ground = p.y;
  float ceiling = 18.0 - p.y;
  float wall1 = p.x + 18.0;
  float wall2 = 18.0 - p.x;
  float wall3 = p.z + 18.0;
  float wall4 = 18.0 - p.z;
  float dist = min(ground, min(ceiling, min(wall1, min(wall2, min(wall3, wall4)))));

  int mat = MATERIAL_NONE;
  if (dist == ground) {
    mat = MATERIAL_GROUND;
  } else if (dist == ceiling) {
    mat = MATERIAL_CEILING;
  } else if (dist == wall1 || dist == wall2) {
    mat = MATERIAL_WALL_ZY;
  } else if (dist == wall3 || dist == wall4) {
    mat = MATERIAL_WALL_XY;
  }

  return DistMat(dist, mat);
}

DistMat sd_lights(in vec3 p)
{
  float light1 = sd_sphere(p - vec3(0.0, 10.0, -18.0), 1.8);
  float light2 = sd_sphere(p - vec3(18.0, 10.0, 0.0), 1.8);
  float light3 = sd_sphere(p - vec3(0.0, 10.0, 18.0), 1.8);
  float light4 = sd_sphere(p - vec3(-18.0, 10.0, 0.0), 1.8);
  float dist = min(light1, min(light2, min(light3, light4)));
  
  int mat = MATERIAL_NONE;
  if (dist == light1) {
    mat = MATERIAL_LIGHT1;
  } else if (dist == light2) {
    mat = MATERIAL_LIGHT2;
  } else if (dist == light3) {
    mat = MATERIAL_LIGHT3;
  } else if (dist == light4) {
    mat = MATERIAL_LIGHT4;
  }

  return DistMat(dist, mat);
}

float sd_water(in vec3 p)
{
  return (abs(p.x) <= POOL_SIZE_X && abs(p.z) <= POOL_SIZE_Z) ? p.y - 3.0 - water_height(p.xz) : INFINITY;
}

float sd_ball(in vec3 p)
{
  return sd_sphere(p - vec3(0.0, 3.0 + 0.1 * sin(2.0 * u_time), 0.0), 2.0);
}

DistMat sd_scene_no_water(in vec3 p)
{
  float pool = sd_pool(p);
  DistMat room = sd_room(p);
  DistMat lights = sd_lights(p);
  float ball = sd_ball(p);
  float dist = min(pool, min(room.dist, min(lights.dist, ball)));

  int mat = MATERIAL_NONE;
  if (dist == pool) {
    mat = MATERIAL_POOL;
  } else if (dist == room.dist) {
    mat = room.mat;
  } else if (dist == lights.dist) {
    mat = lights.mat;
  } else if (dist == ball) {
    mat = MATERIAL_BALL;
  }

  return DistMat(dist, mat);
}

DistMat sd_scene(in vec3 p)
{
  DistMat scene = sd_scene_no_water(p);
  float water = sd_water(p);
  float dist = min(scene.dist, water);

  int mat = MATERIAL_NONE;
  if (dist == scene.dist) {
    mat = scene.mat;
  } else if (dist == water) {
    mat = MATERIAL_WATER;
  }

  return DistMat(dist, mat);
}

/* -------------------------------- Textures -------------------------------- */

// XXX: Sampled textures?

vec3 tile_texture(in vec2 uv)
{
  vec2 q = round(uv);
  return vec3(mix(smoothstep(0.0, 0.07, abs(uv.x - q.x)), smoothstep(0.0, 0.07, abs(uv.y - q.y)), 0.5));
}

vec3 checker_texture(in vec2 uv)
{
  vec2 q = round(uv);
  return vec3(smoothstep(0.0, 0.03, abs(uv.x - q.x) - abs(uv.y - q.y)));
}

/* -------------------------------- Rendering ------------------------------- */

const Light g_lights[] = Light[](
  Light(vec3(0.0, 10.0, -15.0), vec3(0.7)),
  Light(vec3(15.0, 10.0, 0.0), vec3(0.7, 0.2, 0.2)),
  Light(vec3(0.0, 10.0, 15.0), vec3(0.2, 0.7, 0.2)),
  Light(vec3(-15.0, 10.0, 0.0), vec3(0.2, 0.2, 0.7))
);

vec3 render_color(in vec3 p, in vec3 n, in int mat, in vec3 camera)
{
  Light light = g_lights[g_active_light];
  Material m;

  switch (mat) {
    case MATERIAL_POOL:
      m = Material(vec3(0.8), 0.6, 1.0, 0.0, 1.0);
      break;
    case MATERIAL_GROUND:
      m = Material(checker_texture(p.xz), 0.6, 1.0, 0.0, 1.0);
      break;
    case MATERIAL_CEILING:
      m = Material(tile_texture(p.xz), 0.6, 1.0, 0.0, 1.0);
      break;
    case MATERIAL_WALL_XY:
      m = Material(tile_texture(p.xy), 0.6, 1.0, 0.0, 1.0);
      break;
    case MATERIAL_WALL_ZY:
      m = Material(tile_texture(p.zy), 0.6, 1.0, 0.0, 1.0);
      break;
    case MATERIAL_LIGHT1:
      if (g_active_light == 0) return light.color + 0.3;
      m = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      break;
    case MATERIAL_LIGHT2:
      if (g_active_light == 1) return light.color + 0.3;
      m = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      break;
    case MATERIAL_LIGHT3:
      if (g_active_light == 2) return light.color + 0.3;
      m = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      break;
    case MATERIAL_LIGHT4:
      if (g_active_light == 3) return light.color + 0.3;
      m = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      break;
    case MATERIAL_WATER:
      m = Material(vec3(0.0, 0.25, 0.49), 0.6, 1.0, 1.0, 30.0);
      break;
    case MATERIAL_BALL:
      m = Material(vec3(1.0, 0.0, 0.0), 0.3, 0.6, 1.0, 60.0);
      break;
  }

  return calc_color(p, n, camera, light, m);
}

vec3 render(in vec3 camera, in vec3 ray_dir)
{
  DistMat scene = cast_ray(camera, ray_dir);
  if (scene.mat == MATERIAL_NONE) {
    return vec3(0.0);
  }

  Light light = g_lights[g_active_light];
  vec3 p = camera + scene.dist * ray_dir;
  vec3 n = calc_normal(p);
  vec3 color = render_color(p, n, scene.mat, camera);

  if (scene.mat == MATERIAL_WATER) {
    PointMat refl = water_reflection(p, n, camera, light);
    PointMat refr = water_refraction(p, n, camera, light);

    color = 0.2 * color + 0.7 * render_color(refr.p, calc_normal(refr.p), refr.mat, camera);

    vec3 view_dir = normalize(camera - p);
    float fresnel = pow(clamp(1.0 - dot(view_dir, n), 0.0, 1.0), 5.0);

#ifdef REFLECT_ENTIRE_SCENE
    color += 0.1 * fresnel * render_color(refl.p, calc_normal(refl.p), refl.mat, camera);
#else
    if (refl.mat == MATERIAL_BALL) {
      const Material mat = Material(vec3(1.0, 0.0, 0.0), 0.3, 0.6, 1.0, 60.0);
      color += 0.1 * fresnel * calc_color(refl.p, calc_normal(refl.p), camera, light, mat);
    }
#endif
  }

  return color;
}

/* ---------------------------------- Main ---------------------------------- */

void main()
{
  vec2 uv = gl_FragCoord.xy / u_resolution;
  uv = 2.0 * uv - 1.0;
  uv.x *= u_resolution.x / u_resolution.y;
#ifdef CAMERA_MOVEMENT
  vec3 camera = vec3(15.0 * cos(0.3 * u_time), 5.5, 15.0 * sin(0.3 * u_time));
#else
  vec3 camera = vec3(0.0, 5.5, 15.0);
#endif
  vec3 look_at = vec3(0.0, 0.0, 0.0);

  vec3 back = normalize(camera - look_at);
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), back));
  vec3 up = cross(back, right);

  vec3 ray_dir = normalize(uv.x * right + uv.y * up + 1.0 * (-back));
  vec3 color = render(camera, ray_dir);

  color = pow(color, vec3(0.4545));  // Gamma correction

  frag_color = vec4(color, 1.0);
}
