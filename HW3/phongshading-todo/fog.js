// 雾化参数对象
var fogParams = {
    fogColor: [153, 179, 204],  // RGB 0~255
    fogStart: 5.0,              // 线性雾起始距离
    fogEnd: 20.0,               // 线性雾结束距离
    fogDensity: 0.05,           // 指数雾密度
    fogType: 1                  // 雾类型: 0:none, 1:linear, 2:exp, 3:exp2
};

// 创建 dat.GUI
var gui = new dat.GUI();
gui.addColor(fogParams, 'fogColor').name('Fog Color');
gui.add(fogParams, 'fogStart', 0, 50).name('Fog Start');
gui.add(fogParams, 'fogEnd', 0, 100).name('Fog End');
gui.add(fogParams, 'fogDensity', 0, 0.2).step(0.001).name('Fog Density');
gui.add(fogParams, 'fogType', { None: 0, Linear: 1, Exp: 2, Exp2: 3 }).name('Fog Type');
