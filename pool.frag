#version 430

uniform float u_time;
uniform vec2 u_resolution;

out vec4 frag_color;

/* ----------------------- Constants / Types / Globals ---------------------- */

#define CAMERA_MOVEMENT false

struct DistMat
{
  float dist;
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

const int MATERIAL_NONE = 0;
const int MATERIAL_POOL = 1;
const int MATERIAL_GROUND = 2;
const int MATERIAL_WALL_XY = 3;
const int MATERIAL_WALL_ZY = 4;
const int MATERIAL_CEILING = 5;
const int MATERIAL_LIGHTS = 6;

const float POOL_SIZE_X = 8.0;
const float POOL_SIZE_Z = 8.0;

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

vec3 calc_color(in vec3 p, in vec3 n, in vec3 camera, in Light light, in Material mat)
{
  vec3 light_dir = normalize(light.pos - p);
  vec3 view_dir = normalize(camera - p);

  float diff_coeff = max(0.0, dot(light_dir, n));
  float spec_coeff = pow(max(0.0, dot(reflect(-light_dir, n), view_dir)), mat.shininess);

  float ambient = mat.ambient;
  float diffuse = mat.diffuse * diff_coeff;
  float specular = mat.specular * spec_coeff;
  float intensity = ambient + diffuse + specular;

  return intensity * light.color * mat.color;
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
  // dist += 0.002 * noise(30.0 * p);  // Stipple

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

float sd_lights(in vec3 p)
{
  float light1 = sd_sphere(p - vec3(0.0, 10.0, -18.0), 1.8);
  float light2 = sd_sphere(p - vec3(18.0, 10.0, 0.0), 1.8);
  float light3 = sd_sphere(p - vec3(0.0, 10.0, 18.0), 1.8);
  float light4 = sd_sphere(p - vec3(-18.0, 10.0, 0.0), 1.8);
  float dist = min(light1, min(light2, min(light3, light4)));
  // XXX: Return material info?
  return dist;
}

DistMat sd_scene(in vec3 p)
{
  float pool = sd_pool(p);
  DistMat room = sd_room(p);
  float lights = sd_lights(p);
  float dist = min(pool, min(room.dist, lights));

  int mat = MATERIAL_NONE;
  if (dist == pool) {
    mat = MATERIAL_POOL;
  } else if (dist == room.dist) {
    mat = room.mat;
  } else if (dist == lights) {
    mat = MATERIAL_LIGHTS;
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

vec3 render(in vec3 camera, in vec3 ray_dir)
{
  DistMat scene = cast_ray(camera, ray_dir);
  if (scene.mat == MATERIAL_NONE) {
    return vec3(0.0);
  }

  const Light light = Light(vec3(0.0, 10.0, 0.0), vec3(1.0));
  vec3 p = camera + scene.dist * ray_dir;
  vec3 n = calc_normal(p);

  // XXX: Lookup table or conditionals?
#if 0
  Material mats[] = Material[](
    Material(vec3(0.8), 0.6, 1.0, 0.0, 0.0),             // MATERIAL_POOL
    Material(checker_texture(p.xz), 0.6, 1.0, 0.0, 0.0), // MATERIAL_GROUND
    Material(tile_texture(p.xy), 0.6, 1.0, 0.0, 0.0),    // MATERIAL_WALL_XY
    Material(tile_texture(p.zy), 0.6, 1.0, 0.0, 0.0),    // MATERIAL_WALL_ZY
    Material(tile_texture(p.xz), 0.6, 1.0, 0.0, 0.0),    // MATERIAL_CEILING
    Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0)             // MATERIAL_LIGHTS
  );
  return calc_color(p, n, camera, light, mats[scene.mat - 1]);
#else
  Material mat;
  if (scene.mat == MATERIAL_POOL) {
    mat = Material(vec3(0.8), 0.6, 1.0, 0.0, 1.0);
  } else if (scene.mat == MATERIAL_GROUND) {
    mat = Material(checker_texture(p.xz), 0.6, 1.0, 0.0, 1.0);
  } else if (scene.mat == MATERIAL_CEILING) {
    mat = Material(tile_texture(p.xz), 0.6, 1.0, 0.0, 1.0);
  } else if (scene.mat == MATERIAL_WALL_XY) {
    mat = Material(tile_texture(p.xy), 0.6, 1.0, 0.0, 1.0);
  } else if (scene.mat == MATERIAL_WALL_ZY) {
    mat = Material(tile_texture(p.zy), 0.6, 1.0, 0.0, 1.0);
  } else if (scene.mat == MATERIAL_LIGHTS) {
    mat = Material(vec3(1.0), 0.6, 1.0, 1.0, 30.0);
  }
  return calc_color(p, n, camera, light, mat);
#endif
}

/* ---------------------------------- Main ---------------------------------- */

void main()
{
  vec2 uv = gl_FragCoord.xy / u_resolution;
  uv = 2.0 * uv - 1.0;
  uv.x *= u_resolution.x / u_resolution.y;

  vec3 camera = CAMERA_MOVEMENT ?
    vec3(15.0 * cos(0.3 * u_time), 5.5, 15.0 * sin(0.3 * u_time)) :
    vec3(0.0, 5.5, 15.0);
  vec3 look_at = vec3(0.0, 0.0, 0.0);

  vec3 back = normalize(camera - look_at);
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), back));
  vec3 up = cross(back, right);

  vec3 ray_dir = normalize(uv.x * right + uv.y * up + 1.0 * (-back));
  vec3 color = render(camera, ray_dir);

  color = pow(color, vec3(0.4545));  // Gamma correction

  frag_color = vec4(color, 1.0);
}
