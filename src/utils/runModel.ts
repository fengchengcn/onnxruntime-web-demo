import { InferenceSession, Tensor } from 'onnxruntime-web';

/**
 * 环境变量初始化
 */
function init() {
  // env.wasm.simd = false;
}

/**
 * 创建 CPU 推理会话 (使用 WebAssembly)
 * @param model 模型文件的 ArrayBuffer 数据
 */
export async function createModelCpu(model: ArrayBuffer): Promise<InferenceSession> {
  init();
  return await InferenceSession.create(model, { executionProviders: ['wasm'] });
}

/**
 * 创建 GPU 推理会话 (使用 WebGL)
 * @param model 模型文件的 ArrayBuffer 数据
 */
export async function createModelGpu(model: ArrayBuffer): Promise<InferenceSession> {
  init();
  return await InferenceSession.create(model, { executionProviders: ['webgl'] });
}

/**
 * 模型预热：生成随机输入并运行一次推理，以初始化推理引擎的内部缓存
 * @param model 推理会话实例
 * @param dims 输入张量的维度
 */
export async function warmupModel(model: InferenceSession, dims: number[]) {
  // 生成随机输入数据并调用 Session.run() 进行预热查询
  const size = dims.reduce((a, b) => a * b);
  const warmupTensor = new Tensor('float32', new Float32Array(size), dims);

  for (let i = 0; i < size; i++) {
    warmupTensor.data[i] = Math.random() * 2.0 - 1.0;  // 随机值 [-1.0, 1.0)
  }
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = warmupTensor;
    await model.run(feeds);
  } catch (e) {
    console.error(e);
  }
}

/**
 * 运行推理任务
 * @param model 推理会话实例
 * @param preprocessedData 预处理后的输入 Tensor
 * @returns 返回一个包含输出 Tensor 和推理耗时(ms)的元组
 */
export async function runModel(model: InferenceSession, preprocessedData: Tensor): Promise<[Tensor, number]> {
  const start = new Date();
  try {
    const feeds: Record<string, Tensor> = {};
    // 将预处理后的张量映射到模型的第一个输入节点
    feeds[model.inputNames[0]] = preprocessedData;

    // 执行推理
    const outputData = await model.run(feeds);

    const end = new Date();
    // 计算推理时间
    const inferenceTime = (end.getTime() - start.getTime());

    // 获取模型的第一个输出节点的结果
    const output = outputData[model.outputNames[0]];

    return [output, inferenceTime];
  } catch (e) {
    console.error('推理执行出错:', e);
    throw new Error('Inference failed');
  }
}