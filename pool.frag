#version 430

uniform float u_time;
uniform vec2 u_resolution;

out vec4 frag_color;

#define CAMERA_MOVEMENT true

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

const float POOL_SIZE_X = 8.0;
const float POOL_SIZE_Z = 8.0;

float sd_box(in vec3 p, in vec3 s)
{
  vec3 q = abs(p) - s;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
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

DistMat sd_scene(in vec3 p)
{
  float pool = sd_pool(p);
  return DistMat(pool, MATERIAL_POOL);
}

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
  vec3 incident = normalize(p - light.pos);
  vec3 line_of_sight = normalize(p - camera);

  float diff_coeff = max(0.0, dot(-incident, n));
  float spec_coeff = pow(max(0.0, dot(reflect(incident, n), -line_of_sight)), mat.shininess);

  float ambient = mat.ambient;
  float diffuse = mat.diffuse * diff_coeff;
  float specular = mat.specular * spec_coeff;
  float intensity = ambient + diffuse + specular;

  return intensity * light.color * mat.color;
}

vec3 render(in vec3 camera, in vec3 ray_dir)
{
  DistMat scene = cast_ray(camera, ray_dir);
  if (scene.mat == MATERIAL_POOL) {
    const Light light = Light(vec3(0.0, 10.0, 0.0), vec3(1.0));
    const Material mat = Material(vec3(0.8), 0.6, 1.0, 0.0, 0.0);
    vec3 p = camera + scene.dist * ray_dir;
    vec3 n = calc_normal(p);
    return calc_color(p, n, camera, light, mat);
  }
  return vec3(0.0);
}

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
