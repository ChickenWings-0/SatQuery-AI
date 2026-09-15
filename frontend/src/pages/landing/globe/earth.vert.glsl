varying vec3 vNormal;
varying vec3 vWorld;
varying vec3 vLocal;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vLocal = position;
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}
