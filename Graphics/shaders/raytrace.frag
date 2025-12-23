#version 300 es
precision highp float;

out vec4 FragColor;

in vec2 vUv;

uniform vec2  uResolution;
uniform vec3  uCameraPos;
uniform vec3  uCameraFront;
uniform vec3  uCameraRight;
uniform vec3  uCameraUp;
uniform float uFov;
uniform float uTime;

// 三个球心（可由 CPU 实时修改，实现交互拖动）
uniform vec3  uSphereCenters[3];
// 当前选中的球索引（-1 表示没有选中，仅用于高亮）
uniform int   uSelectedIndex;

// -------------------------------
// 简单场景：三个球 + 地面平面 + 一个点光源
// 支持：Phong 光照 + 阴影 + 反射/折射 + 雾 + 稍微的凹凸扰动
// -------------------------------

struct HitInfo {
    float t;
    vec3  pos;
    vec3  normal;
    int   materialID;
};

// 材质类型
const int MAT_DIFFUSE  = 0;
const int MAT_MIRROR   = 1;
const int MAT_GLASS    = 2;
const int MAT_FRACTAL  = 3;
const int MAT_PERSON   = 4; // 方块人

// 伪随机
float hash(vec2 p)
{
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p.x + p.y) * 43758.5453);
}

// 球求交
bool intersectSphere(vec3 ro, vec3 rd, vec3 center, float radius,
                     out float tHit, out vec3 normal)
{
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    tHit = -b - h;
    if (tHit < 0.0) tHit = -b + h;
    if (tHit < 0.0) return false;
    vec3 p = ro + rd * tHit;
    normal = normalize(p - center);
    return true;
}

// 无限平面 y = h（向上法线）
bool intersectPlane(vec3 ro, vec3 rd, float h,
                    out float tHit, out vec3 normal)
{
    if (abs(rd.y) < 1e-4) return false;
    float t = (h - ro.y) / rd.y;
    if (t < 0.0) return false;
    tHit = t;
    normal = vec3(0.0, 1.0, 0.0);
    return true;
}

// 轴对齐包围盒（AABB）求交，用于方块人的身体/四肢
bool intersectAABB(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax,
                   out float tHit, out vec3 normal)
{
    vec3 invD = 1.0 / rd;
    vec3 t0s = (bmin - ro) * invD;
    vec3 t1s = (bmax - ro) * invD;

    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger  = max(t0s, t1s);

    float tmin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);

    if (tmax < 0.0 || tmin > tmax) return false;

    tHit = tmin;
    if (tHit < 0.0) {
        tHit = tmax;
        if (tHit < 0.0) return false;
    }

    // 根据命中的面估计法线
    vec3 hitPos = ro + rd * tHit;
    const float eps = 1e-3;
    if (abs(hitPos.x - bmin.x) < eps) normal = vec3(-1.0, 0.0, 0.0);
    else if (abs(hitPos.x - bmax.x) < eps) normal = vec3(1.0, 0.0, 0.0);
    else if (abs(hitPos.y - bmin.y) < eps) normal = vec3(0.0, -1.0, 0.0);
    else if (abs(hitPos.y - bmax.y) < eps) normal = vec3(0.0, 1.0, 0.0);
    else if (abs(hitPos.z - bmin.z) < eps) normal = vec3(0.0, 0.0, -1.0);
    else                                  normal = vec3(0.0, 0.0, 1.0);

    return true;
}

// -------------------------------
// 分形 SDF：简单 Mandelbulb 距离场
// -------------------------------

float mandelbulbDE(vec3 p)
{
    // 基于常见的 Mandelbulb 距离估计公式，迭代次数和幂次可根据性能调整
    vec3 z = p;
    float dr = 1.0;
    float r = 0.0;

    const int ITER = 8;
    const float POWER = 8.0;

    for (int i = 0; i < ITER; ++i)
    {
        r = length(z);
        if (r > 2.5) break;

        // 转为球坐标
        float theta = acos(z.z / max(r, 1e-6));
        float phi   = atan(z.y, z.x);

        dr = pow(r, POWER - 1.0) * POWER * dr + 1.0;

        float zr = pow(r, POWER);
        theta *= POWER;
        phi   *= POWER;

        // 回到笛卡尔坐标
        z = zr * vec3(
            sin(theta) * cos(phi),
            sin(theta) * sin(phi),
            cos(theta)
        ) + p;
    }

    return 0.5 * log(r) * r / dr;
}

// 计算分形 SDF 对应的法线（梯度近似）
vec3 fractalNormal(vec3 p)
{
    float eps = 0.001;
    float dx = mandelbulbDE(p + vec3(eps, 0.0, 0.0)) - mandelbulbDE(p - vec3(eps, 0.0, 0.0));
    float dy = mandelbulbDE(p + vec3(0.0, eps, 0.0)) - mandelbulbDE(p - vec3(0.0, eps, 0.0));
    float dz = mandelbulbDE(p + vec3(0.0, 0.0, eps)) - mandelbulbDE(p - vec3(0.0, 0.0, eps));
    return normalize(vec3(dx, dy, dz));
}

// 对 Mandelbulb 做简单的 ray marching
bool intersectFractal(vec3 ro, vec3 rd, out float tHit, out vec3 normal)
{
    float t = 0.0;
    const float MAX_DIST = 25.0;
    const int   MAX_STEPS = 100;
    const float SURF_EPS = 0.001;

    // 将分形放在稍远处，避免与现有物体过于重叠
    // 可以通过平移 ro 来实现：p = 原始点 - fractalCenter
    vec3 center = vec3(-2.5, 0.5, -4.5);

    for (int i = 0; i < MAX_STEPS; ++i)
    {
        vec3 pos = ro + rd * t;
        vec3 q = pos - center;
        float d = mandelbulbDE(q);
        if (d < SURF_EPS)
        {
            tHit = t;
            normal = fractalNormal(q);
            return true;
        }
        t += d;
        if (t > MAX_DIST) break;
    }

    return false;
}

// 计算场景最近求交
bool intersectScene(vec3 ro, vec3 rd, out HitInfo hit)
{
    hit.t = 1e20;
    hit.materialID = -1;

    float t; vec3 n;

    // 球 1：蓝色漫反射
    if (intersectSphere(ro, rd, uSphereCenters[0], 0.8, t, n) && t < hit.t) {
        hit.t = t;
        hit.pos = ro + rd * t;
        hit.normal = n;
        hit.materialID = MAT_DIFFUSE;
    }

    // 原“球 2”改为方块人，由多个 AABB 组合而成
    {
        // 以 uSphereCenters[1] 的 xz 为中心，将 y 放在地面之上
        vec3 base = vec3(uSphereCenters[1].x, -0.5, uSphereCenters[1].z);

        // 身体
        vec3 bodyMin = base + vec3(-0.25, 0.0, -0.15);
        vec3 bodyMax = base + vec3( 0.25, 0.9,  0.15);

        // 头
        vec3 headMin = base + vec3(-0.2, 0.9, -0.2);
        vec3 headMax = base + vec3( 0.2, 1.3,  0.2);

        // 左腿 / 右腿
        vec3 legLMin = base + vec3(-0.18, -0.6, -0.12);
        vec3 legLMax = base + vec3(-0.02,  0.0,  0.12);
        vec3 legRMin = base + vec3( 0.02, -0.6, -0.12);
        vec3 legRMax = base + vec3( 0.18,  0.0,  0.12);

        // 左臂 / 右臂
        vec3 armLMin = base + vec3(-0.45, 0.4, -0.10);
        vec3 armLMax = base + vec3(-0.25, 0.9,  0.10);
        vec3 armRMin = base + vec3( 0.25, 0.4, -0.10);
        vec3 armRMax = base + vec3( 0.45, 0.9,  0.10);

        float tb; vec3 nb;

        if (intersectAABB(ro, rd, bodyMin, bodyMax, tb, nb) && tb < hit.t) {
            hit.t = tb;
            hit.pos = ro + rd * tb;
            hit.normal = nb;
            hit.materialID = MAT_PERSON;
        }
        if (intersectAABB(ro, rd, headMin, headMax, tb, nb) && tb < hit.t) {
            hit.t = tb;
            hit.pos = ro + rd * tb;
            hit.normal = nb;
            hit.materialID = MAT_PERSON;
        }
        if (intersectAABB(ro, rd, legLMin, legLMax, tb, nb) && tb < hit.t) {
            hit.t = tb;
            hit.pos = ro + rd * tb;
            hit.normal = nb;
            hit.materialID = MAT_PERSON;
        }
        if (intersectAABB(ro, rd, legRMin, legRMax, tb, nb) && tb < hit.t) {
            hit.t = tb;
            hit.pos = ro + rd * tb;
            hit.normal = nb;
            hit.materialID = MAT_PERSON;
        }
        if (intersectAABB(ro, rd, armLMin, armLMax, tb, nb) && tb < hit.t) {
            hit.t = tb;
            hit.pos = ro + rd * tb;
            hit.normal = nb;
            hit.materialID = MAT_PERSON;
        }
        if (intersectAABB(ro, rd, armRMin, armRMax, tb, nb) && tb < hit.t) {
            hit.t = tb;
            hit.pos = ro + rd * tb;
            hit.normal = nb;
            hit.materialID = MAT_PERSON;
        }
    }

    // 球 3：玻璃
    if (intersectSphere(ro, rd, uSphereCenters[2], 0.6, t, n) && t < hit.t) {
        hit.t = t;
        hit.pos = ro + rd * t;
        hit.normal = n;
        hit.materialID = MAT_GLASS;
    }

    // 地面平面 y = -0.5，棋盘纹理
    if (intersectPlane(ro, rd, -0.5, t, n) && t < hit.t) {
        hit.t = t;
        hit.pos = ro + rd * t;
        hit.normal = n;
        hit.materialID = MAT_DIFFUSE;
    }

    // 分形几何（Mandelbulb），使用 ray marching
    if (intersectFractal(ro, rd, t, n) && t < hit.t) {
        hit.t = t;
        hit.pos = ro + rd * t;
        hit.normal = n;
        hit.materialID = MAT_FRACTAL;
    }

    return (hit.materialID != -1);
}

// 计算棋盘纹理颜色（代替漫反射贴图）
vec3 checkerColor(vec3 pos)
{
    float scale = 2.0;
    float cx = floor(pos.x * scale);
    float cz = floor(pos.z * scale);
    float c = mod(cx + cz, 2.0);
    return mix(vec3(0.9), vec3(0.1), c);
}

// 计算单个交点的局部颜色（不包含反射/折射）
vec3 computeLocalColor(HitInfo hit, vec3 rd)
{
    // 点光源位置
    vec3 lightPos = vec3(2.5, 3.0, -1.0);
    vec3 lightColor = vec3(1.0, 0.95, 0.9);

    // 确定基础颜色
    vec3 baseColor;
    if (hit.materialID == MAT_DIFFUSE) {
        // 地面用棋盘；球用固定色
        if (abs(hit.normal.y - 1.0) < 0.5 && hit.pos.y < -0.49) {
            baseColor = checkerColor(hit.pos);
        } else {
            baseColor = vec3(0.2, 0.6, 0.9);
        }
    } else if (hit.materialID == MAT_PERSON) {
        // 简单“我的世界”风格方块人配色：
        // 头部：肤色；身体：蓝绿色衣服；裤子：深蓝；手脚：肤色
        if (hit.pos.y > -0.5 + 0.9) {
            // 头部区域
            baseColor = vec3(0.95, 0.8, 0.6);
        } else if (hit.pos.y > -0.5 && hit.pos.y <= -0.5 + 0.9) {
            // 身体和手臂
            baseColor = vec3(0.3, 0.7, 0.85);
        } else {
            // 腿部
            baseColor = vec3(0.1, 0.2, 0.5);
        }
    } else if (hit.materialID == MAT_MIRROR) {
        baseColor = vec3(0.9);
    } else if (hit.materialID == MAT_GLASS) {
        baseColor = vec3(0.9, 0.98, 1.0);
    } else { // MAT_FRACTAL
        // 分形采用偏发光的绿色调，便于在场景中突出显示
        baseColor = vec3(0.1, 0.9, 0.4);
    }

    // 法线凹凸扰动（模拟凹凸贴图）
    float bump = 0.15 * sin(hit.pos.x * 5.0) * sin(hit.pos.z * 5.0);
    vec3 perturbedNormal = normalize(hit.normal + bump * vec3(0.3, 1.0, 0.2));

    // 阴影测试
    vec3 lightDir = lightPos - hit.pos;
    float lightDist = length(lightDir);
    lightDir /= lightDist;

    // shadow ray
    HitInfo shadowHit;
    bool inShadow = false;
    if (intersectScene(hit.pos + perturbedNormal * 0.01, lightDir, shadowHit)) {
        if (shadowHit.t < lightDist) {
            inShadow = true;
        }
    }

    vec3 ambient = 0.05 * baseColor;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    if (!inShadow) {
        float diff = max(dot(perturbedNormal, lightDir), 0.0);
        vec3 viewDir = normalize(-rd);
        vec3 reflectDir = reflect(-lightDir, perturbedNormal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

        diffuse = diff * baseColor * lightColor;
        specular = 0.4 * spec * lightColor;
    }

    return ambient + diffuse + specular;
}

// 使用迭代方式追踪光线（避免递归）
vec3 shade(vec3 ro, vec3 rd)
{
    const int MAX_BOUNCES = 3;
    vec3 throughput = vec3(1.0);
    vec3 color = vec3(0.0);
    
    vec3 currentRo = ro;
    vec3 currentRd = rd;
    
    for (int bounce = 0; bounce <= MAX_BOUNCES; ++bounce) {
        HitInfo hit;
        if (!intersectScene(currentRo, currentRd, hit)) {
            // 背景
            float t = 0.5 * (currentRd.y + 1.0);
            vec3 skyTop = vec3(0.1, 0.2, 0.5);
            vec3 skyBottom = vec3(0.6, 0.7, 0.9);
            color += throughput * mix(skyBottom, skyTop, t);
            break;
        }
        
        // 计算局部颜色
        vec3 localColor = computeLocalColor(hit, currentRd);
        color += throughput * localColor;
        
        // 如果是漫反射或分形，停止追踪
        if (hit.materialID == MAT_DIFFUSE || hit.materialID == MAT_FRACTAL) {
            break;
        }
        
        vec3 viewDir = normalize(-currentRd);
        float bump = 0.15 * sin(hit.pos.x * 5.0) * sin(hit.pos.z * 5.0);
        vec3 perturbedNormal = normalize(hit.normal + bump * vec3(0.3, 1.0, 0.2));
        
        if (hit.materialID == MAT_MIRROR) {
            // 镜面反射
            vec3 reflDir = reflect(-viewDir, perturbedNormal);
            currentRo = hit.pos + perturbedNormal * 0.01;
            currentRd = reflDir;
            throughput *= 0.8;
        }
        else if (hit.materialID == MAT_GLASS) {
            // 玻璃：折射和反射
            float cosi = clamp(dot(viewDir, perturbedNormal), -1.0, 1.0);
            float etai = 1.0;
            float etat = 1.5;
            vec3 n = perturbedNormal;
            if (cosi < 0.0) {
                cosi = -cosi;
            } else {
                // inside
                float tmp = etai;
                etai = etat;
                etat = tmp;
                n = -n;
            }
            float eta = etai / etat;
            float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
            
            vec3 reflDir = reflect(-viewDir, perturbedNormal);
            if (k < 0.0) {
                // 全反射
                currentRo = hit.pos + n * 0.01;
                currentRd = reflDir;
            } else {
                // Fresnel 近似
                float R0 = pow((etai - etat) / (etai + etat), 2.0);
                float c = 1.0 - cosi;
                float fresnel = R0 + (1.0 - R0) * c * c * c * c * c;
                
                // 根据Fresnel选择反射或折射
                if (fresnel > 0.5) {
                    currentRo = hit.pos + n * 0.01;
                    currentRd = reflDir;
                    throughput *= fresnel;
                } else {
                    vec3 refrDir = normalize(eta * viewDir - (eta * cosi + sqrt(k)) * n);
                    currentRo = hit.pos - n * 0.01;
                    currentRd = refrDir;
                    throughput *= (1.0 - fresnel);
                }
            }
        }
        
        // 如果throughput太小，停止追踪
        if (max(max(throughput.x, throughput.y), throughput.z) < 0.01) {
            break;
        }
    }
    
    return color;
}

void main()
{
    // 简单多重采样抗锯齿 + 景深
    // 为了保持实时帧率，采样数不宜太高，这里用 4 次抖动采样
    int SAMPLES = 4;
    vec3 accumColor = vec3(0.0);

    float aspect = uResolution.x / uResolution.y;
    float tanFov = tan(radians(uFov) * 0.5);

    // 景深参数：焦平面距离和光圈大小（可根据需要微调）
    float focusDist = 3.0;
    float aperture  = 0.03;

    for (int i = 0; i < SAMPLES; ++i)
    {
        // 生成子像素抖动（抗锯齿随机偏移）
        float jx = hash(vUv + vec2(float(i), 0.123));
        float jy = hash(vUv + vec2(0.456, float(i)));
        vec2 jitter = vec2(jx, jy) - 0.5;

        vec2 uv = (vUv + jitter / uResolution) * 2.0 - 1.0;
        uv.x *= aspect;

        vec3 rayDirCamera = normalize(vec3(uv.x * tanFov, uv.y * tanFov, -1.0));
        vec3 rayDirWorld =
            normalize(rayDirCamera.x * uCameraRight +
                      rayDirCamera.y * uCameraUp +
                      rayDirCamera.z * (-uCameraFront));

        vec3 ro = uCameraPos;
        vec3 rd = rayDirWorld;

        // 景深：从镜头光圈随机采样一个起点，使焦平面外略微虚化
        // 采样单位圆盘
        float r1 = hash(vUv + vec2(float(i) * 2.0, uTime));
        float r2 = hash(vUv + vec2(uTime, float(i) * 3.0));
        float theta = 2.0 * 3.1415926 * r1;
        float r = sqrt(r2);
        vec2 disk = r * vec2(cos(theta), sin(theta)) * aperture;

        vec3 focusPoint = ro + rd * focusDist;
        vec3 newOrigin = ro + disk.x * uCameraRight + disk.y * uCameraUp;
        vec3 newDir = normalize(focusPoint - newOrigin);

        accumColor += shade(newOrigin, newDir);
    }

    vec3 color = accumColor / float(SAMPLES);

    // 如果物体被选中，稍微提亮高光，作为简单高亮效果
    // 这里通过再次快速查询当前像素射线是否打到选中球，来增强其亮度
    if (uSelectedIndex >= 0 && uSelectedIndex < 3)
    {
        vec2 uvCenter = vUv * 2.0 - 1.0;
        uvCenter.x *= aspect;
        vec3 rdCam = normalize(vec3(uvCenter.x * tanFov, uvCenter.y * tanFov, -1.0));
        vec3 rdWorld =
            normalize(rdCam.x * uCameraRight +
                      rdCam.y * uCameraUp +
                      rdCam.z * (-uCameraFront));

        float tHit; vec3 n;
        bool hitSel = false;
        if (uSelectedIndex == 0)
            hitSel = intersectSphere(uCameraPos, rdWorld, uSphereCenters[0], 0.8, tHit, n);
        else if (uSelectedIndex == 1)
            hitSel = intersectSphere(uCameraPos, rdWorld, uSphereCenters[1], 0.8, tHit, n);
        else if (uSelectedIndex == 2)
            hitSel = intersectSphere(uCameraPos, rdWorld, uSphereCenters[2], 0.6, tHit, n);

        if (hitSel)
        {
            color *= 1.15;
        }
    }

    // 屏幕中央绿色十字准星（基于屏幕像素坐标）
    {
        vec2 pixel = vUv * uResolution;
        vec2 center = 0.5 * uResolution;
        vec2 d = abs(pixel - center);

        float halfLength = 8.0;  // 准星臂长（像素）
        float halfThickness = 1.0; // 线条粗细（像素）

        // 竖线条件：x 接近中心且 y 在一定范围内
        float vertLine = step(d.x, halfThickness) * step(d.y, halfLength);
        // 横线条件：y 接近中心且 x 在一定范围内
        float horizLine = step(d.y, halfThickness) * step(d.x, halfLength);

        float crossMask = clamp(vertLine + horizLine, 0.0, 1.0);

        vec3 crossColor = vec3(0.1, 1.0, 0.1); // 亮绿色
        float crossAlpha = 0.85;

        color = mix(color, crossColor, crossMask * crossAlpha);
    }

    FragColor = vec4(color, 1.0);
}
