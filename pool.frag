
#extension GL_EXT_gpu_shader4 : enable

uniform float u_time;
uniform vec2 u_resolution;

#define CAMERA_MOVEMENT 1

/* ---------------------------- General Utilities --------------------------- */

const float INFINITY = 1.0 / 0.0;

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

float sd_scene(in vec3 p, out int mat, in bool enable_water = true, in bool enable_active_light = true);
vec3 render_color(in vec3 p, in vec3 n, in int mat, in vec3 camera, in Light light);

float shadow(in vec3 p, in vec3 incident)
{
  const float min_t = 0.1;
  const float max_t = 10.0;
  float t = min_t;
  float result = 1.0;

  for (int i = 0; i < 64 && t <= max_t; i++) {
    vec3 q = p + t * (-incident);
    int mat;
    float dist = sd_scene(q, mat, false, false);
    result = min(result, 16.0 * dist / t);
    if (result < -1.0) break;
    t += clamp(dist, 0.1, 10.0);
  }

  result = max(result, -1.0);
  return 0.25 * (1.0 + result) * (1.0 + result) * (2.0 - result);
}

vec3 calc_color(in vec3 p, in vec3 n, in vec3 camera, in Light light, in Material mat)
{
  vec3 incident = normalize(p - light.pos);
  vec3 line_of_sight = normalize(p - camera);

  float ambient = mat.ambient;
  float diffuse = mat.diffuse * max(0.0, dot(-incident, n));
  float specular = mat.specular * pow(max(0.0, dot(reflect(incident, n), -line_of_sight)), mat.shininess);
  float intensity = ambient + diffuse + specular;
  intensity *= shadow(p, incident);

  return intensity * light.color * mat.color;
}

vec3 calc_normal(in vec3 p)
{
  const vec2 eps = vec2(0.001, 0.0);
  int mat;
  vec3 n = vec3(
    sd_scene(p + eps.xyy, mat) - sd_scene(p - eps.xyy, mat),
    sd_scene(p + eps.yxy, mat) - sd_scene(p - eps.yxy, mat),
    sd_scene(p + eps.yyx, mat) - sd_scene(p - eps.yyx, mat)
  );
  return normalize(n);
}

bool cast_ray(in vec3 origin, in vec3 dir, out float hit_t, out int mat)
{
  const float min_t = 0.01;
  const float max_t = 50.0;
  float t = min_t;

  for (int i = 0; i < 256 && t <= max_t; i++) {
    vec3 p = origin + t * dir;
    float dist = sd_scene(p, mat);
    if (dist < 0.001) {
      hit_t = t;
      return true;
    }
    t += dist;
  }

  return false;
}

vec3 reflection(in vec3 p, in vec3 n, in vec3 camera, in Light light, bool enable_water = false)
{
  const float max_t = 50.0;
  const float min_t = 0.01;
  float t = min_t;

  vec3 incident = normalize(p - light.pos);
  vec3 ray_dir = reflect(incident, n);

  for (int i = 0; i < 256 && t <= max_t; i++) {
    int mat;
    vec3 q = 0.002 * n + p + t * ray_dir;
    float dist = sd_scene(q, mat, enable_water);
    if (dist < 0.001) {
      return render_color(q, calc_normal(q), mat, camera, light);
    }
    t += dist;
  }

  return vec3(0.0);
}

vec3 refraction(in vec3 p, in vec3 n, in vec3 camera, in Light light, float eta)
{
  const float max_t = 10.0;
  const float min_t = 0.01;
  float t = min_t;

  vec3 line_of_sight = normalize(p - camera);
  vec3 ray_dir = refract(line_of_sight, n, eta);

  for (int i = 0; i < 256 && t <= max_t; i++) {
    int mat;
    vec3 q = -0.002 * n + p + t * ray_dir;
    float dist = sd_scene(q, mat, false, false);
    if (dist < 0.001) {
      return render_color(q, calc_normal(q), mat, camera, light);
    }
    t += dist;
  }

  return vec3(0.0);
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

vec4 water_height_d(in vec2 p)
{
  float value = 0.0;
  vec3 grad = vec3(0.0);
  float freq = 1.0;
  float amp = 1.0;

  for (int i = 0; i < 3; i++) {
    vec4 n = noise_d(vec3(freq * p + 0.8 * u_time, 0.8 * u_time));
    value += amp * n.x;
    grad += amp * freq * n.yzw;
    freq *= 2.0;
    amp /= 2.0;
  }

  return vec4(value, grad) / 20.0;
}

vec3 water_normal(in vec2 p)
{
  vec3 grad = water_height_d(p).yzw;
  vec3 n = vec3(-grad.x, 1.0, -grad.y);
  return normalize(n);
}

/* -------------------------------- Textures -------------------------------- */

vec3 tile_texture(in vec2 uv)
{
  vec2 q = round(uv);
  return vec3(mix(smoothstep(0.0, 0.07, abs(uv.x - q.x)), smoothstep(0.0, 0.07, abs(uv.y - q.y)), 0.5));
}

vec3 tile_texture_sampled(in vec2 uv, in vec2 duvdx, in vec2 duvdy)
{
  const int max_samples = 10;

  int x_steps = 1 + int(clamp(10.0 * length(duvdx), 0.0, float(max_samples - 1)));
  int y_steps = 1 + int(clamp(10.0 * length(duvdy), 0.0, float(max_samples - 1)));

  vec3 total = vec3(0.0);

  for (int i = 0; i < x_steps; i++) {
    for (int j = 0; j < y_steps; j++) {
      vec2 st = vec2(i, j) / vec2(x_steps, y_steps);
      total += tile_texture(uv + st.x * duvdx + st.y * duvdy);
    }
  }

  return total / (float(x_steps) * float(y_steps));
}

vec3 checker_texture(in vec2 uv)
{
  vec2 q = round(uv);
  return vec3(smoothstep(0.0, 0.03, abs(uv.x - q.x) - abs(uv.y - q.y)));
}

vec3 checker_texture_sampled(in vec2 uv, in vec2 duvdx, in vec2 duvdy)
{
  const int max_samples = 10;

  int x_steps = 1 + int(clamp(10.0 * length(duvdx), 0.0, float(max_samples - 1)));
  int y_steps = 1 + int(clamp(10.0 * length(duvdy), 0.0, float(max_samples - 1)));

  vec3 total = vec3(0.0);

  for (int i = 0; i < x_steps; i++) {
    for (int j = 0; j < y_steps; j++) {
      vec2 st = vec2(i, j) / vec2(x_steps, y_steps);
      total += checker_texture(uv + st.x * duvdx + st.y * duvdy);
    }
  }

  return total / (float(x_steps) * float(y_steps));
}

/* ---------------------------------- SDFs ---------------------------------- */

float sd_box(in vec3 p, in vec3 s)
{
  vec3 q = abs(p) - s;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sd_sphere(in vec3 p, float r)
{
  return length(p) - r;
}

float sd_plane(in vec3 p, in vec3 n, in float h)
{
  return dot(p, n) + h;
}

const int MATERIAL_NONE = 0;
const int MATERIAL_WATER = 1;
const int MATERIAL_POOL = 2;
const int MATERIAL_BALL = 3;
const int MATERIAL_GROUND = 4;
const int MATERIAL_XY_WALL = 5;
const int MATERIAL_ZY_WALL = 6;
const int MATERIAL_LIGHT1 = 7;
const int MATERIAL_LIGHT2 = 8;
const int MATERIAL_LIGHT3 = 9;
const int MATERIAL_LIGHT4 = 10;

const float POOL_SIZE_X = 8.0;
const float POOL_SIZE_Z = 8.0;

int g_active_light = int(0.5 * u_time) % 4;

float sd_pool(in vec3 p)
{
  float edge1 = sd_box(p - vec3(-POOL_SIZE_X, -1.2, 0.0), vec3(0.3, 1.8, POOL_SIZE_Z + 0.3));
  float edge2 = sd_box(p - vec3(POOL_SIZE_X, -1.2, 0.0), vec3(0.3, 1.8, POOL_SIZE_Z + 0.3));
  float edge3 = sd_box(p - vec3(0.0, -1.2, -POOL_SIZE_Z), vec3(POOL_SIZE_X + 0.3, 1.8, 0.3));
  float edge4 = sd_box(p - vec3(0.0, -1.2, POOL_SIZE_Z), vec3(POOL_SIZE_X + 0.3, 1.8, 0.3));
  float bottom = sd_box(p - vec3(0.0, -3.0, 0.0), vec3(POOL_SIZE_X, 0.1, POOL_SIZE_Z));
  float dist = min(edge1, min(edge2, min(edge3, min(edge4, bottom))));
  dist += 0.002 * noise(30.0 * p);  // Stipple
  return dist;
}

float sd_room(in vec3 p, out int mat)
{
  float wall1 = sd_plane(p, vec3(0.0, 0.0, 1.0), 18.0);
  float wall2 = sd_plane(p, vec3(0.0, 0.0, -1.0), 18.0);
  float wall3 = sd_plane(p, vec3(1.0, 0.0, 0.0), 18.0);
  float wall4 = sd_plane(p, vec3(-1.0, 0.0, 0.0), 18.0);
  float ground = sd_plane(p, vec3(0.0, 1.0, 0.0), 3.0);

  float dist = min(wall1, min(wall2, min(wall3, min(wall4, ground))));

  if (dist == ground) {
    mat = MATERIAL_GROUND;
  } else if (dist == wall1 || dist == wall2) {
    mat = MATERIAL_XY_WALL;
  } else if (dist == wall3 || dist == wall4) {
    mat = MATERIAL_ZY_WALL;
  }

  return dist;
}

float sd_water(in vec3 p)
{
  return (abs(p.x) < POOL_SIZE_X && abs(p.z) < POOL_SIZE_Z) ? p.y - water_height(p.xz) : INFINITY;
}

float sd_lights(in vec3 p, out int mat, in bool enable_active_light)
{
  float light1 = sd_sphere(p - vec3(0.0, 10.0, -18.0), 1.8);
  float light2 = sd_sphere(p - vec3(18.0, 10.0, 0.0), 1.8);
  float light3 = sd_sphere(p - vec3(0.0, 10.0, 18.0), 1.8);
  float light4 = sd_sphere(p - vec3(-18.0, 10.0, 0.0), 1.8);

  float dist;
  if (enable_active_light) {
    dist = min(light1, min(light2, min(light3, light4)));
  } else if (g_active_light == 0) {
    dist = min(light2, min(light3, light4));
  } else if (g_active_light == 1) {
    dist = min(light1, min(light3, light4));
  } else if (g_active_light == 2) {
    dist = min(light1, min(light2, light4));
  } else if (g_active_light == 3) {
    dist = min(light1, min(light2, light3));
  }

  if (dist == light1) {
    mat = MATERIAL_LIGHT1;
  } else if (dist == light2) {
    mat = MATERIAL_LIGHT2;
  } else if (dist == light3) {
    mat = MATERIAL_LIGHT3;
  } else if (dist == light4) {
    mat = MATERIAL_LIGHT4;
  }

  return dist;
}

float sd_scene(in vec3 p, out int mat, in bool enable_water = true, in bool enable_active_light = true)
{
  float pool = sd_pool(p);
  float dist = pool;

  int room_mat;
  float room = sd_room(p, room_mat);
  dist = min(dist, room);

  float ball = sd_sphere(p - vec3(0.0, 0.0, 0.0), 2.0);
  dist = min(dist, ball);

  float water = enable_water ? sd_water(p) : INFINITY;
  dist = min(dist, water);

  int light_mat;
  float lights = sd_lights(p, light_mat, enable_active_light);
  dist = min(dist, lights);

  if (dist == pool) {
    mat = MATERIAL_POOL;
  } else if (dist == room) {
    mat = room_mat;
  } else if (dist == ball) {
    mat = MATERIAL_BALL;
  } else if (dist == water) {
    mat = MATERIAL_WATER;
  } else if (dist == lights) {
    mat = light_mat;
  }

  return dist;
}

/* -------------------------------- Rendering ------------------------------- */

vec3 render_color(in vec3 p, in vec3 n, in int mat, in vec3 camera, in Light light)
{
  switch (mat) {
    case MATERIAL_WATER: {
      const Material mat = Material(vec3(0.0, 0.25, 0.49), 0.6, 1.0, 1.0, 30.0);
      vec3 color = calc_color(p, n, camera, light, mat);
      return mix(vec3(1.0), color, n.y);
    }
    case MATERIAL_POOL: {
      const Material mat = Material(vec3(0.8), 0.6, 1.0, 0.0, 0.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_BALL: {
      const Material mat = Material(vec3(1.0, 0.0, 0.0), 0.3, 0.6, 1.0, 60.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_GROUND: {
      vec3 tex = checker_texture_sampled(p.xz, dFdx(p.xz), dFdy(p.xz));
      Material mat = Material(tex, 0.6, 1.0, 0.0, 0.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_XY_WALL: {
      vec3 tex = tile_texture_sampled(p.xy, dFdx(p.xy), dFdy(p.xy));
      Material mat = Material(tex, 0.6, 1.0, 0.0, 0.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_ZY_WALL: {
      vec3 tex = tile_texture_sampled(p.zy, dFdx(p.zy), dFdy(p.zy));
      Material mat = Material(tex, 0.6, 1.0, 0.0, 0.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_LIGHT1: {
      if (g_active_light == 0) return vec3(light.color + 0.3);
      const Material mat = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_LIGHT2: {
      if (g_active_light == 1) return vec3(light.color + 0.3);
      const Material mat = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_LIGHT3: {
      if (g_active_light == 2) return vec3(light.color + 0.3);
      const Material mat = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      return calc_color(p, n, camera, light, mat);
    }
    case MATERIAL_LIGHT4: {
      if (g_active_light == 3) return vec3(light.color + 0.3);
      const Material mat = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
      return calc_color(p, n, camera, light, mat);
    }
  }

  return vec3(0.0);
}

Light get_active_light()
{
  switch (g_active_light) {
    case 0:
      return Light(vec3(0.0, 10.0, -15.0), vec3(0.7));
    case 1:
      return Light(vec3(15.0, 10.0, 0.0), vec3(0.7, 0.2, 0.2));
    case 2:
      return Light(vec3(0.0, 10.0, 15.0), vec3(0.2, 0.7, 0.2));
    case 3:
      return Light(vec3(-15.0, 10.0, 0.0), vec3(0.2, 0.2, 0.7));
  }
}

vec3 render(in vec3 camera, in vec3 ray_dir)
{
  float hit_t;
  int mat;

  if (!cast_ray(camera, ray_dir, hit_t, mat)) {
    return vec3(0.0);
  }

  Light light = get_active_light();
  vec3 p = camera + hit_t * ray_dir;
  vec3 n = (mat == MATERIAL_WATER) ? water_normal(p.xz) : calc_normal(p);
  vec3 color = render_color(p, n, mat, camera, light);

  // Handle reflection/refraction.
  switch (mat) {
    case MATERIAL_WATER: {
      vec3 line_of_sight = normalize(p - camera);
      float fresnel = max(0.0, dot(-line_of_sight, n));
      color = 0.2 * color +
        0.7 * refraction(p, n, camera, light, 1.0 / 1.33) +
        0.1 * fresnel * reflection(p, n, camera, light);
      break;
    }
    case MATERIAL_LIGHT1:
    case MATERIAL_LIGHT2:
    case MATERIAL_LIGHT3:
    case MATERIAL_LIGHT4:
      if (!(mat == MATERIAL_LIGHT1 && g_active_light == 0 ||
            mat == MATERIAL_LIGHT2 && g_active_light == 1 ||
            mat == MATERIAL_LIGHT3 && g_active_light == 2 ||
            mat == MATERIAL_LIGHT4 && g_active_light == 3)) {
        color = 0.8 * color +
          0.1 * refraction(p, n, camera, light, 1.0 / 1.52) +
          0.1 * reflection(p, n, camera, light, true);
      }
      break;
  }

  return color;
}

/* ---------------------------------- Main ---------------------------------- */

void main()
{
  vec2 uv = gl_FragCoord.xy / u_resolution;
  uv = 2.0 * uv - 1.0;
  uv.x *= u_resolution.x / u_resolution.y;

#if CAMERA_MOVEMENT
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
  
  gl_FragColor = vec4(color, 1.0);
}
