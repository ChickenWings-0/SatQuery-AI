precision highp float;

uniform vec3 uGround;
uniform vec3 uGrid;
uniform vec3 uLand;
uniform vec3 uRim;
uniform float uGridAlpha;
uniform float uRimStrength;
uniform float uTheme;
uniform vec3 uSun;
uniform sampler2D uLandMask;
uniform float uHasMask;
uniform float uTime;

varying vec3 vNormal;
varying vec3 vWorld;
varying vec3 vLocal;

const float PI = 3.141592653589793;

// A hairline at every `step` degrees, antialiased by screen-space derivative.
float gridLine(float coordDeg, float stepDeg, float width) {
  float f = abs(fract(coordDeg / stepDeg - 0.5) - 0.5) * stepDeg;
  float d = fwidth(coordDeg) * width;
  return 1.0 - smoothstep(0.0, d, f);
}

// A stipple: a dot field in lat/lon so land reads as points, not a fill.
float stipple(vec2 deg, float cell) {
  vec2 g = fract(deg / cell) - 0.5;
  float r = length(g);
  float d = fwidth(r) * 1.5;
  return 1.0 - smoothstep(0.16, 0.16 + d, r);
}

void main() {
  vec3 n = normalize(vLocal);
  float lat = degrees(asin(clamp(n.y, -1.0, 1.0)));
  float lon = degrees(atan(-n.z, n.x));

  // The graticule: every 10°, heavier every 30°.
  float minor = max(gridLine(lat, 10.0, 1.0), gridLine(lon, 10.0, 1.0));
  float major = max(gridLine(lat, 30.0, 1.4), gridLine(lon, 30.0, 1.4));
  float grid = max(minor * uGridAlpha, major * uGridAlpha * 2.0);

  // Land: sample the mask (equirectangular), stipple it.
  float land = 0.0;
  if (uHasMask > 0.5) {
    vec2 uv = vec2(0.5 + lon / 360.0, 0.5 + lat / 180.0);
    float m = texture2D(uLandMask, uv).r;
    land = step(0.5, m) * stipple(vec2(lon, lat), 1.2);
  }

  // Day / night: the sun direction rotates slowly; nothing goes black.
  float sun = dot(normalize(vWorld), normalize(uSun));
  float day = smoothstep(-0.15, 0.25, sun);
  float landAlpha = mix(0.35, 1.0, day) * 0.9;

  // Fresnel rim: the atmosphere.
  vec3 view = normalize(cameraPosition - vWorld);
  float fres = pow(1.0 - max(dot(normalize(vNormal), view), 0.0), 3.0);

  vec3 color = uGround;
  color = mix(color, uGrid, grid);
  color = mix(color, uLand, land * landAlpha);
  color += uRim * fres * uRimStrength;

  // On the light theme the sphere is a shade below the paper and the grid is
  // darker than its ground rather than lighter; the uniforms carry that.
  float alpha = 1.0;
  gl_FragColor = vec4(color, alpha);
}
