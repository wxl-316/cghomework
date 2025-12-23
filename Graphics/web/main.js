const vertexSource = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aPos;

out vec2 vUv;

void main()
{
    vUv = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const fragmentSource = `#version 300 es
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

// 更复杂光源：3 个点光源，由 CPU 传入位置和颜色
uniform vec3  uLightPos[3];
uniform vec3  uLightColor[3];

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

//**********
// 封闭房间尺寸：宽=6，高=3，深=6，中心在原点
const vec3 ROOM_MIN = vec3(-12.0, -2.0, -12.0);
const vec3 ROOM_MAX = vec3( 12.0, 10.0,  12.0);

// 返回 6 个面中最近的一次命中，并附带材质 ID
bool intersectRoom(vec3 ro, vec3 rd,
                   out float tHit, out vec3 normal, out int matID)
{
    // 先按 AABB 求交
    vec3 invD = 1.0 / rd;
    vec3 t0   = (ROOM_MIN - ro) * invD;
    vec3 t1   = (ROOM_MAX - ro) * invD;
    vec3 tsm  = min(t0, t1);
    vec3 tbg  = max(t0, t1);
    float tmn = max(tsm.x, max(tsm.y, tsm.z));
    float tmx = min(tbg.x, min(tbg.y, tbg.z));
    if (tmx < 0.0 || tmn > tmx) return false;
    tHit = tmn < 0.0 ? tmx : tmn;
    if (tHit < 0.0) return false;

    // 计算命中点 & 面法线
    vec3 hitP = ro + rd * tHit;
    const float eps = 1e-3;
    // 将法线设为指向房间内部（这样从房间内部观察时法线方向与光照一致）
    if      (abs(hitP.x - ROOM_MIN.x) < eps) { normal = vec3( 1,0,0); matID = MAT_DIFFUSE; }
    else if (abs(hitP.x - ROOM_MAX.x) < eps) { normal = vec3(-1,0,0); matID = MAT_DIFFUSE; }
    else if (abs(hitP.y - ROOM_MIN.y) < eps) { normal = vec3(0, 1,0); matID = MAT_DIFFUSE; } // 地板（朝上）
    else if (abs(hitP.y - ROOM_MAX.y) < eps) { normal = vec3(0,-1,0); matID = MAT_DIFFUSE; } // 天花板（朝下）
    else if (abs(hitP.z - ROOM_MIN.z) < eps) { normal = vec3(0,0, 1); matID = MAT_MIRROR;  } // -Z 墙 镜面（法线朝 +Z，面向房间内部）
    else                                     { normal = vec3(0,0,-1); matID = MAT_DIFFUSE; } // +Z 墙
    return true;
}
//**********

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

    // 将分形放在房间中的一个角落（现在与其它物体一起按正方形四顶点排列）
    // 可以通过平移 ro 来实现：p = 原始点 - fractalCenter
    // 把分形中心也放低，靠近地面
    vec3 center = vec3(-3.0, -0.5 + 0.05, 3.0);

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

    // 新增：封闭房间
    int mat;
    if (intersectRoom(ro, rd, t, n, mat) && t < hit.t) {
        hit.t = t;
        hit.pos = ro + rd * t;
        hit.normal = n;
        hit.materialID = mat;
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
    // -------------------------------
    // 基础颜色（由材质决定）
    // -------------------------------
    vec3 baseColor;
    if (hit.materialID == MAT_DIFFUSE) {
        // 地面用棋盘；球用固定色
        if (abs(hit.normal.y - 1.0) < 0.5 && hit.pos.y < -0.49) {
            // 地面：使用棋盘纹理并稍微提高亮度
            baseColor = checkerColor(hit.pos) * 1.2;
            baseColor = min(baseColor, vec3(1.0));
        }
        // 墙面：法线主要在 X 或 Z 方向，设置为灰白色
        else if (abs(hit.normal.x) > 0.9 || abs(hit.normal.z) > 0.9) {
            // 三面墙：橙黄色（更暖、更明显）
            baseColor = vec3(1.0, 0.65, 0.2);
        } else {
            // 其他漫反射物体（例如球）保持原来配色
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

    // -------------------------------
    // 半球环境光（天空/地面不同颜色）
    // -------------------------------
    vec3 hemiSky    = vec3(0.25, 0.35, 0.7);
    vec3 hemiGround = vec3(0.15, 0.10, 0.05);
    float ny = clamp(perturbedNormal.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 hemi = mix(hemiGround, hemiSky, ny);

    vec3 ambient  = 0.25 * baseColor * hemi;
    vec3 diffuse  = vec3(0.0);
    vec3 specular = vec3(0.0);

    // 视线方向
    vec3 viewDir = normalize(-rd);

    // -------------------------------
    // 光源 0：主光（暖色，带阴影）
    // -------------------------------
    {
        vec3 lightPos  = uLightPos[0];
        vec3 lightCol  = uLightColor[0];
        vec3 lightVec  = lightPos - hit.pos;
        float lightDist = length(lightVec);
        vec3 lightDir  = lightVec / max(lightDist, 1e-4);

        // shadow ray
        HitInfo shadowHit;
        bool inShadow = false;
        if (intersectScene(hit.pos + perturbedNormal * 0.01, lightDir, shadowHit)) {
            if (shadowHit.t < lightDist) {
                inShadow = true;
            }
        }

        if (!inShadow) {
            float diff = max(dot(perturbedNormal, lightDir), 0.0);
            vec3 reflectDir = reflect(-lightDir, perturbedNormal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

            // 距离衰减（近处更亮，远处更暗）
            float att = 1.0 / (1.0 + 0.15 * lightDist + 0.05 * lightDist * lightDist);

            diffuse  += att * diff * baseColor * lightCol;
            specular += att * 0.5 * spec * lightCol;
        }
    }

    // -------------------------------
    // 光源 1：冷色填充光（无阴影）
    // -------------------------------
    {
        vec3 lightPos  = uLightPos[1];
        vec3 lightCol  = uLightColor[1];
        vec3 lightVec  = lightPos - hit.pos;
        float lightDist = length(lightVec);
        vec3 lightDir  = lightVec / max(lightDist, 1e-4);

        float diff = max(dot(perturbedNormal, lightDir), 0.0);
        vec3 reflectDir = reflect(-lightDir, perturbedNormal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);

        float att = 1.0 / (1.0 + 0.2 * lightDist + 0.1 * lightDist * lightDist);

        diffuse  += 0.7 * att * diff * baseColor * lightCol;
        specular += 0.3 * att * spec * lightCol;
    }

    // -------------------------------
    // 光源 2：随时间旋转的彩色边缘光（无阴影）
    // -------------------------------
    {
        vec3 lightPos  = uLightPos[2];
        vec3 lightCol  = uLightColor[2];
        vec3 lightVec  = lightPos - hit.pos;
        float lightDist = length(lightVec);
        vec3 lightDir  = lightVec / max(lightDist, 1e-4);

        float diff = max(dot(perturbedNormal, lightDir), 0.0);
        vec3 halfVec = normalize(lightDir + viewDir);
        float spec = pow(max(dot(perturbedNormal, halfVec), 0.0), 24.0);

        float att = 1.0 / (1.0 + 0.1 * lightDist + 0.07 * lightDist * lightDist);

        diffuse  += 0.5 * att * diff * baseColor * lightCol;
        specular += 0.6 * att * spec * lightCol;
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
        // 计算用于着色的扰动法线，但对于镜面/玻璃不应使用扰动法线来计算反射/折射
        vec3 perturbedNormalForShading = normalize(hit.normal + bump * vec3(0.3, 1.0, 0.2));
        vec3 perturbedNormal = perturbedNormalForShading;
        if (hit.materialID == MAT_MIRROR || hit.materialID == MAT_GLASS) {
            // 对镜面和玻璃使用真实法线以获得清晰反射/折射
            perturbedNormal = hit.normal;
        }
        
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
    int SAMPLES = 8;
    vec3 accumColor = vec3(0.0);

    float aspect = uResolution.x / uResolution.y;
    float tanFov = tan(radians(uFov) * 0.5);

    // 景深参数：焦平面距离和光圈大小（可根据需要微调）
    float focusDist = 3.0;
    // 为提高清晰度将光圈设为 0（关闭景深），必要时可改小值比如 0.005
    float aperture  = 0.01;

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
        float aspect = uResolution.x / uResolution.y;
        float tanFov = tan(radians(uFov) * 0.5);

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
`;

// ---------------- WebGL2 初始化与主循环（JavaScript 部分） ----------------

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('glcanvas'));
/** @type {WebGL2RenderingContext} */
const gl = canvas.getContext('webgl2');

if (!gl) {
  alert('当前浏览器不支持 WebGL2，请使用现代浏览器（Chrome/Edge/Firefox 等）。');
  throw new Error('WebGL2 not supported');
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = Math.floor(canvas.clientWidth * dpr);
  const displayHeight = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    throw new Error('Shader compile error');
  }
  return shader;
}

function createProgram(vsSource, fsSource) {
  const vs = compileShader(gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    throw new Error('Program link error');
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

// 创建全屏三角形
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
const vertices = new Float32Array([
  -1.0, -1.0,
   3.0, -1.0,
  -1.0,  3.0,
]);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 2 * 4, 0);
gl.bindVertexArray(null);

// 创建着色器程序
const program = createProgram(vertexSource, fragmentSource);
gl.useProgram(program);

// uniform 位置
const locResolution = gl.getUniformLocation(program, 'uResolution');
const locCamPos     = gl.getUniformLocation(program, 'uCameraPos');
const locCamFront   = gl.getUniformLocation(program, 'uCameraFront');
const locCamRight   = gl.getUniformLocation(program, 'uCameraRight');
const locCamUp      = gl.getUniformLocation(program, 'uCameraUp');
const locFov        = gl.getUniformLocation(program, 'uFov');
const locTime       = gl.getUniformLocation(program, 'uTime');
const locLightPos   = gl.getUniformLocation(program, 'uLightPos');
const locLightColor = gl.getUniformLocation(program, 'uLightColor');
const locSphere0    = gl.getUniformLocation(program, 'uSphereCenters[0]');
const locSphere1    = gl.getUniformLocation(program, 'uSphereCenters[1]');
const locSphere2    = gl.getUniformLocation(program, 'uSphereCenters[2]');
const locSelected   = gl.getUniformLocation(program, 'uSelectedIndex');

// 调试：只打印一次，检查 sphereCenters 与 uniform 位置是否正常
let _debugPrinted = false;

// 相机与交互（用 JS 简化版 Camera）
let camPos   = [0, 1.5, 2.0];
let yaw      = -90; // 朝 -Z
let pitch    = -10;
let fovDeg   = 60;

// 与 GLSL 中房间尺寸保持一致（已放大 4 倍）
const ROOM_MIN_JS = [-12.0, -2.0, -12.0];
const ROOM_MAX_JS = [ 12.0, 10.0,  12.0];

// 简单“玩家”物理，用于更贴近《我的世界》的第一人称手感：
// - 固定在地面上移动（WASD 只在水平面移动）
// - 重力 + 跳跃（Space）
// - Shift 潜行减速，Ctrl 冲刺加速
const GROUND_Y = -0.5;      // 与场景中地面平面一致
const PLAYER_HEIGHT = 2.0;  // 眼睛高度（地面到相机的距离）
let playerY = GROUND_Y + PLAYER_HEIGHT;
let velY    = 0.0;
let onGround = true;

/** @type {Record<string, boolean>} */
const keys = {};

canvas.addEventListener('click', () => {
  canvas.requestPointerLock?.();
});

document.addEventListener('pointerlockchange', () => {
  // 可以根据是否锁定来调整 UI 或敏感度，这里保持简单
});

document.addEventListener('mousemove', (e) => {
    if (document.pointerLockElement !== canvas) return;
    // 降低鼠标灵敏度，避免旋转过快
    const sensitivity = 0.04;
    const dx = e.movementX * sensitivity;
    const dy = -e.movementY * sensitivity;
  yaw += dx;
  pitch += dy;
  pitch = Math.max(-89, Math.min(89, pitch));
});

document.addEventListener('keydown', (e) => {
  keys[e.code] = true;
  // 数字键 1/2/3 切换选中球
  if (e.code === 'Digit1') selectedSphere = 0;
  if (e.code === 'Digit2') selectedSphere = 1;
  if (e.code === 'Digit3') selectedSphere = 2;
});
document.addEventListener('keyup', (e) => {
  keys[e.code] = false;
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  // 调整滚轮方向：向上滚动应视角放大（FOV 变小）
  fovDeg += e.deltaY * 0.02;
  fovDeg = Math.max(20, Math.min(90, fovDeg));
}, { passive: false });

// 球心（包含方块人的“基准点”）
let sphereCenters = [
    // 第一球半径 0.8：中心 y = GROUND_Y + 0.8
    [-3.0, GROUND_Y + 0.8, -3.0],
    // 方块人（以 xz 为基准，垂直位置由方块人逻辑决定），这里 y 也设为接近地面
    [ 3.0, GROUND_Y + 0.1, -3.0],
    // 第三球半径 0.6：中心 y = GROUND_Y + 0.6
    [ 3.0, GROUND_Y + 0.6,  3.0],
];

let selectedSphere = -1;

function normalize3(v) {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len === 0) return [0, 0, 0];
  return [v[0]/len, v[1]/len, v[2]/len];
}

function cross(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
}

let lastTime = performance.now() / 1000;

function update(dt) {
  // 根据 yaw/pitch 计算相机方向
  const cy = Math.cos(yaw * Math.PI / 180);
  const sy = Math.sin(yaw * Math.PI / 180);
  const cp = Math.cos(pitch * Math.PI / 180);
  const sp = Math.sin(pitch * Math.PI / 180);

  const front = normalize3([cy * cp, sp, sy * cp]);
  const worldUp = [0, 1, 0];
  const right = normalize3(cross(front, worldUp));
  const up = normalize3(cross(right, front)); // 标准右手系，上方向保持与世界坐标一致

  // ----------- 玩家竖直方向物理（重力 + 跳跃） -----------
  /*const GRAVITY = 9.8;
  const JUMP_SPEED = 5.0;

  // Space 跳跃（仅在落地时生效）
  if (keys['Space'] && onGround) {
    velY = JUMP_SPEED;
    onGround = false;
  }

  // 应用重力
  velY -= GRAVITY * dt;
  playerY += velY * dt;

  const minPlayerY = GROUND_Y + PLAYER_HEIGHT;
  if (playerY <= minPlayerY) {
    playerY = minPlayerY;
    velY = 0.0;
    onGround = true;
  }*/

  // ----------- 水平移动（WASD，只在 XZ 平面） -----------
  const frontHoriz = normalize3([front[0], 0, front[2]]);
  //let rightHoriz = cross(frontHoriz, worldUp);
  //rightHoriz = normalize3(rightHoriz);
  const rightHoriz = normalize3(cross(frontHoriz, worldUp));
  let moveDir = [0, 0, 0];

  // 基础移动速度 + 潜行 / 冲刺
  let speed = 3.0;
  //if (keys['ShiftLeft']) speed *= 0.4; // 潜行：更慢
  //if (keys['ControlLeft']) speed *= 1.8; // 冲刺：更快

  if (keys['KeyW']) {
    moveDir[0] += frontHoriz[0];
    moveDir[2] += frontHoriz[2];
  }
  if (keys['KeyS']) {
    moveDir[0] -= frontHoriz[0];
    moveDir[2] -= frontHoriz[2];
  }
  if (keys['KeyA']) {
    moveDir[0] -= rightHoriz[0];
    moveDir[2] -= rightHoriz[2];
  }
  if (keys['KeyD']) {
    moveDir[0] += rightHoriz[0];
    moveDir[2] += rightHoriz[2];
  }

  const len = Math.hypot(moveDir[0], moveDir[1], moveDir[2]);
  if (len > 0.0001) {
    moveDir[0] /= len;
    moveDir[1] /= len;
    moveDir[2] /= len;

    const vel = speed * dt;
    camPos[0] += moveDir[0] * vel;
    camPos[2] += moveDir[2] * vel;
  }

  /* 相机高度由玩家竖直位置决定，避免“飞行模式”
  camPos[1] = playerY;

  return { front, right, up };*/
  // ----------- 新增：Z/C 自由升降高度 -----------
    const liftSpeed = 2.0;    // 升降速度（单位：世界坐标/秒）
    // 按 Z 提升高度，按 C 降低高度（用户要求）
    if (keys['KeyZ']) camPos[1] += liftSpeed * dt;
    if (keys['KeyC']) camPos[1] -= liftSpeed * dt;

  // 可选：给一个合理的上下限，防止穿到地心或飞太高
  camPos[1] = Math.max(-1.0, Math.min(8.0, camPos[1]));

    // 将相机限制在房间内部，避免移出房间（使用与 GLSL 中对应的房间边界）
    const _eps = 0.1;
    camPos[0] = Math.max(ROOM_MIN_JS[0] + _eps, Math.min(ROOM_MAX_JS[0] - _eps, camPos[0]));
    camPos[2] = Math.max(ROOM_MIN_JS[2] + _eps, Math.min(ROOM_MAX_JS[2] - _eps, camPos[2]));
    camPos[1] = Math.max(ROOM_MIN_JS[1] + PLAYER_HEIGHT, Math.min(ROOM_MAX_JS[1] - _eps, camPos[1]));

    return { front, right, up };
}

function render(timeMs) {
  resizeCanvas();
  const timeSec = timeMs / 1000;
  const dt = timeSec - lastTime;
  lastTime = timeSec;

  const { front, right, up } = update(dt);

  gl.clearColor(0.1, 0.1, 0.15, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  gl.useProgram(program);
  gl.bindVertexArray(vao);

  gl.uniform2f(locResolution, canvas.width, canvas.height);
  gl.uniform3f(locCamPos, camPos[0], camPos[1], camPos[2]);
  gl.uniform3f(locCamFront, front[0], front[1], front[2]);
  gl.uniform3f(locCamRight, right[0], right[1], right[2]);
  gl.uniform3f(locCamUp,    up[0],    up[1],    up[2]);
  gl.uniform1f(locFov, fovDeg);
  gl.uniform1f(locTime, timeSec);

  // 设置复杂光源：一个主光 + 一个冷色填充光 + 一个随时间旋转的彩色光
  const lightPosArray = new Float32Array(9);
  const lightColorArray = new Float32Array(9);

  // 光源 0：暖色主光，位于右上方，带阴影（在 GLSL 中使用）
    // 提升主光高度并增强强度，使房间整体更亮
    lightPosArray[0] = 0.0; lightPosArray[1] = 5.0; lightPosArray[2] = 0.0;
    lightColorArray[0] = 1.6; lightColorArray[1] = 1.45; lightColorArray[2] = 1.3;

  // 光源 1：冷色填充光，在场景左上
  lightPosArray[3] = -3.0; lightPosArray[4] = 2.0; lightPosArray[5] = -2.0;
  lightColorArray[3] = 0.6; lightColorArray[4] = 0.8; lightColorArray[5] = 1.0;

  // 光源 2：随时间绕场景旋转的彩色边缘光
  const r = 3.0;
  const ang = timeSec * 0.6;
  lightPosArray[6] = Math.cos(ang) * r;
  lightPosArray[7] = 1.8 + Math.sin(timeSec * 0.5) * 0.5;
  lightPosArray[8] = -2.5 + Math.sin(ang) * r;
  lightColorArray[6] = 1.0; lightColorArray[7] = 0.4; lightColorArray[8] = 0.8;

  gl.uniform3fv(locLightPos, lightPosArray);
  gl.uniform3fv(locLightColor, lightColorArray);

    // 强制把三个物体贴合地面，避免因手动修改导致悬空
    sphereCenters[0][1] = GROUND_Y + 0.8; // 球1 半径 0.8
    sphereCenters[1][1] = GROUND_Y + 0.0; // 方块人基准 y 由场景逻辑决定，但这里设为地面高度
    sphereCenters[2][1] = GROUND_Y + 0.6; // 球2 半径 0.6

    if (!_debugPrinted) {
        // 打印 JS-side 数值（展开）
        console.log('DEBUG sphereCenters (JS):', sphereCenters.map(v=>('['+v.map(x=>x.toFixed(3)).join(', ')+']')));
        console.log('DEBUG uniform locs: locSphere0=', locSphere0, 'locSphere1=', locSphere1, 'locSphere2=', locSphere2);
        // 从 GPU 读取回传的 uniform 值并以数组形式打印（展开具体数值）
        const u0 = gl.getUniform(program, locSphere0);
        const u1 = gl.getUniform(program, locSphere1);
        const u2 = gl.getUniform(program, locSphere2);
        console.log('DEBUG read back uniforms from GPU:', Array.from(u0).map(x=>x.toFixed(3)), Array.from(u1).map(x=>x.toFixed(3)), Array.from(u2).map(x=>x.toFixed(3)));
        _debugPrinted = true;
    }

    gl.uniform3f(locSphere0,
        sphereCenters[0][0],
        sphereCenters[0][1],
        sphereCenters[0][2]
    );
    gl.uniform3f(locSphere1,
        sphereCenters[1][0],
        sphereCenters[1][1],
        sphereCenters[1][2]
    );
    gl.uniform3f(locSphere2,
        sphereCenters[2][0],
        sphereCenters[2][1],
        sphereCenters[2][2]
    );

  gl.uniform1i(locSelected, selectedSphere);
    // 读取 shader 中的 uniform 值以确认已正确上传（只打印一次）
    if (!_debugPrinted) {
        const u0 = gl.getUniform(program, locSphere0);
        const u1 = gl.getUniform(program, locSphere1);
        const u2 = gl.getUniform(program, locSphere2);
        console.log('DEBUG read back uniforms from GPU:', u0, u1, u2);
    }

  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindVertexArray(null);

  requestAnimationFrame(render);
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();
requestAnimationFrame(render);


