export const knowledgeTreePresets = {
  '人教A版': {
    id: 'renjiao-a',
    name: '人教A版',
    children: [
      {
        id: 'required-1',
        name: '必修第一册',
        children: [
          { id: 'r1-c1', name: '第一章 集合与常用逻辑用语', topics: ['集合的概念', '集合的运算', '充分条件与必要条件', '全称量词与存在量词'] },
          { id: 'r1-c2', name: '第二章 一元二次函数、方程和不等式', topics: ['等式性质与不等式性质', '基本不等式', '二次函数与一元二次方程、不等式'] },
          { id: 'r1-c3', name: '第三章 函数的概念与性质', topics: ['函数的概念及其表示', '函数的基本性质', '幂函数'] },
          { id: 'r1-c4', name: '第四章 指数函数与对数函数', topics: ['指数', '指数函数', '对数', '对数函数', '函数的应用'] },
          { id: 'r1-c5', name: '第五章 三角函数', topics: ['任意角和弧度制', '三角函数的概念', '诱导公式', '三角函数的图象与性质', '三角恒等变换'] }
        ]
      },
      {
        id: 'required-2',
        name: '必修第二册',
        children: [
          { id: 'r2-c6', name: '第六章 平面向量及其应用', topics: ['平面向量的概念', '平面向量的运算', '平面向量基本定理及坐标表示', '平面向量的应用'] },
          { id: 'r2-c7', name: '第七章 复数', topics: ['复数的概念', '复数的四则运算'] },
          { id: 'r2-c8', name: '第八章 立体几何初步', topics: ['基本立体图形', '立体图形的直观图', '简单几何体的表面积与体积', '空间点、直线、平面之间的位置关系'] }
        ]
      },
      {
        id: 'elective-1',
        name: '选择性必修第一册',
        children: [
          { id: 'e1-c1', name: '第一章 空间向量与立体几何', topics: ['空间向量及其运算', '空间向量的应用'] },
          { id: 'e1-c2', name: '第二章 直线和圆的方程', topics: ['直线的倾斜角与斜率', '直线的方程', '直线的交点坐标与距离公式', '圆的方程', '直线与圆、圆与圆的位置关系'] },
          { id: 'e1-c3', name: '第三章 圆锥曲线的方程', topics: ['椭圆', '双曲线', '抛物线', '直线与圆锥曲线的位置关系'] }
        ]
      },
      {
        id: 'elective-2',
        name: '选择性必修第二册',
        children: [
          { id: 'e2-c4', name: '第四章 数列', topics: ['数列的概念', '等差数列', '等比数列', '数学归纳法'] },
          { id: 'e2-c5', name: '第五章 一元函数的导数及其应用', topics: ['导数的概念及其意义', '导数的运算', '导数在研究函数中的应用'] }
        ]
      },
      {
        id: 'elective-3',
        name: '选择性必修第三册',
        children: [
          { id: 'e3-c6', name: '第六章 计数原理', topics: ['分类加法计数原理与分步乘法计数原理', '排列与组合', '二项式定理'] },
          { id: 'e3-c7', name: '第七章 随机变量及其分布', topics: ['条件概率与全概率公式', '离散型随机变量及其分布列', '离散型随机变量的数字特征', '二项分布与超几何分布', '正态分布'] },
          { id: 'e3-c8', name: '第八章 成对数据的统计分析', topics: ['成对数据的统计相关性', '一元线性回归模型及其应用', '列联表与独立性检验'] }
        ]
      }
    ]
  },
  '高考新题型': {
    id: 'gaokao-new',
    name: '高考新题型',
    children: [
      { id: 'gk-c1', name: '第一章 集合', topics: [] },
      { id: 'gk-c2', name: '第二章 函数', topics: [] },
      { id: 'gk-c3', name: '第三章 导数及其应用', topics: [] },
      { id: 'gk-c4', name: '第四章 三角函数与解三角形', topics: [] },
      { id: 'gk-c5', name: '第五章 平面向量', topics: [] },
      { id: 'gk-c6', name: '第六章 数列', topics: [] },
      { id: 'gk-c7', name: '第七章 不等式、推理与证明', topics: [] },
      { id: 'gk-c8', name: '第八章 空间向量与立体几何', topics: [] },
      { id: 'gk-c9', name: '第九章 解析几何', topics: [] },
      { id: 'gk-c10', name: '第十章 计数原理', topics: [] },
      { id: 'gk-c11', name: '第十一章 概率与统计', topics: [] },
      { id: 'gk-c12', name: '第十二章 算法初步、复数', topics: [] },
      { id: 'gk-c13', name: '第十三章 选讲部分', topics: [] }
    ]
  }
};

export const defaultKnowledgeTree = knowledgeTreePresets['人教A版'];

export const saveCustomKnowledgeTree = (tree) => {
  localStorage.setItem('custom_knowledge_tree', JSON.stringify(tree));
};

export const loadCustomKnowledgeTree = () => {
  const saved = localStorage.getItem('custom_knowledge_tree');
  return saved ? JSON.parse(saved) : null;
};
