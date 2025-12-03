#version 300 es
precision mediump float;

out vec4 FragColor;

uniform float ambientStrength, specularStrength, diffuseStrength, shininess;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec4 FragPosLightSpace;

uniform vec3 viewPos;
uniform vec4 u_lightPosition;
uniform vec3 lightColor;

uniform sampler2D diffuseTexture;
uniform sampler2D depthTexture;
uniform samplerCube cubeSampler;

uniform vec3 fogColor;      // 雾的颜色
uniform float fogStart;     // 雾开始距离（线性雾）
uniform float fogEnd;       // 雾结束距离（线性雾）
uniform float fogDensity;   // 雾浓度（指数雾）
uniform int fogType;


// =================== shadow with 3x3 PCF ===================
float shadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 ||
       projCoords.y < 0.0 || projCoords.y > 1.0)
        return 0.0;

    float currentDepth = projCoords.z;

    // improved bias
    float ndotl = max(dot(normal, lightDir), 0.0);
    float bias = max(0.002 * tan(acos(ndotl)), 0.0005);
    bias = clamp(bias, 0.0, 0.01);

    float shadow = 0.0;
    float texelSize = 1.0 / float(textureSize(depthTexture, 0));

    for(int x = -2; x <= 2; ++x)
    {
        for(int y = -2; y <= 2; ++y)
        {
            float pcfDepth = texture(depthTexture, projCoords.xy + vec2(x, y) * texelSize).r;
            if(currentDepth - bias > pcfDepth)
                shadow += 1.0;
        }
    }
    shadow /= 25.0;

    return shadow;
}

void main()
{
    vec3 TextureColor = texture(diffuseTexture, TexCoord).rgb;

    vec3 norm = normalize(Normal);
    vec3 lightDir = (u_lightPosition.w == 1.0) ?
        normalize(u_lightPosition.xyz - FragPos) :
        normalize(u_lightPosition.xyz);

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir);

    // --------- Phong ---------
    vec3 ambient = ambientStrength * lightColor;

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diffuseStrength * diff * lightColor;

    float spec = 0.0;
    if(diff > 0.0)
        spec = pow(max(dot(norm, halfDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    // lighting sum
    vec3 lit = ambient + diffuse + specular;

    // --------- shadow ---------
    float shadow = shadowCalculation(FragPosLightSpace, norm, lightDir);

    vec3 resultColor = (ambient + (1.0 - shadow) * (diffuse + specular)) * TextureColor;

    //=============== Fog Calculation ===============
    float distance = length(viewPos - FragPos);
    float fogFactor = 1.0;

    if (fogType == 1) {
        // Linear Fog
        fogFactor = (fogEnd - distance) / (fogEnd - fogStart);
    }
    else if (fogType == 2) {
        // Exponential Fog
        fogFactor = exp(-fogDensity * distance);
    }
    else if (fogType == 3) {
        // Exponential-Squared Fog
        fogFactor = exp(-pow(fogDensity * distance, 2.0));
    }

    fogFactor = clamp(fogFactor, 0.0, 1.0);

    // 颜色混合
    vec3 finalColor = mix(fogColor, resultColor, fogFactor);

    FragColor = vec4(finalColor, 1.0);

}
