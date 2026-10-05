import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getTopicLocation, getPlaygroundInfo, getPracticeKernelInfo, PRACTICE_DIR, PlaygroundType } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, chapterId, tier, target = 'solution', code, playgroundType = 'cpp', volumeId, practiceKernel, file } = body;

    // 1. Create New Practice Kernel (kernels/practice/*.cu)
    if (type === 'create_practice_kernel') {
      const rawName = body.filename || body.name || '';
      const sanitized = rawName.trim().replace(/[^a-zA-Z0-9_\-\.]/g, '_');
      if (!sanitized) {
        return NextResponse.json({ success: false, error: 'Please provide a valid kernel filename.' }, { status: 400 });
      }
      const cleanName = sanitized.endsWith('.cu') ? sanitized : `${sanitized}.cu`;
      const filePath = path.join(PRACTICE_DIR, cleanName);
      if (fs.existsSync(filePath)) {
        return NextResponse.json({ success: false, error: `Kernel "${cleanName}" already exists.` }, { status: 400 });
      }

      const defaultTemplate = body.code || `// kernels/practice/${cleanName}
// Open Practice CUDA Kernel
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \\
            exit(1); \\
        } \\
    } while (0)

__global__ void custom_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Write your custom logic here
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    float *h_in = new float[N];
    float *h_out = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    custom_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    std::cout << "[PASSED] Kernel ${cleanName} executed successfully! Sample output: " << h_out[0] << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    return 0;
}
`;

      fs.writeFileSync(filePath, defaultTemplate, 'utf-8');
      return NextResponse.json({
        success: true,
        filename: cleanName,
        relPath: path.join('kernels', 'practice', cleanName),
        message: `Successfully created ${cleanName}`,
      });
    }

    // 2. Delete Practice Kernel
    if (type === 'delete_practice_kernel') {
      const rawName = body.filename || body.name || '';
      const cleanName = rawName.endsWith('.cu') ? rawName : `${rawName}.cu`;
      const filePath = path.join(PRACTICE_DIR, cleanName);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        return NextResponse.json({ success: true, filename: cleanName });
      }
      return NextResponse.json({ success: false, error: 'File not found' }, { status: 404 });
    }

    // 3. Save Practice Kernel (kernels/practice/*.cu)
    if (type === 'practice') {
      const targetFile = practiceKernel || file || body.filename || 'gpu_check.cu';
      const kInfo = getPracticeKernelInfo(targetFile);
      if (!kInfo) {
        return NextResponse.json({ success: false, error: `Practice kernel ${targetFile} not found` }, { status: 404 });
      }
      fs.writeFileSync(kInfo.filePath, code, 'utf-8');
      return NextResponse.json({ success: true, path: kInfo.filename, practiceKernel: kInfo.filename });
    }

    // 4. Save Playground (C++, CUDA, Kernel)
    if (type === 'playground') {
      const pgInfo = getPlaygroundInfo(playgroundType as PlaygroundType);
      fs.writeFileSync(pgInfo.filePath, code, 'utf-8');
      return NextResponse.json({ success: true, path: pgInfo.filename, playgroundType: pgInfo.type });
    }

    // 2. Fork to Sandbox Action
    if (type === 'fork_to_sandbox') {
      const targetPg: PlaygroundType = body.isKernel ? 'kernel' : body.isCuda ? 'cuda' : 'cpp';
      const pgInfo = getPlaygroundInfo(targetPg);

      // Safeguard: backup existing playground content if non-empty
      if (fs.existsSync(pgInfo.filePath)) {
        const existing = fs.readFileSync(pgInfo.filePath, 'utf-8');
        if (existing.trim() && existing !== code) {
          const backupPath = pgInfo.filePath.replace(/\.(cpp|cu)$/, '.backup.$1');
          fs.writeFileSync(backupPath, existing, 'utf-8');
        }
      }

      fs.writeFileSync(pgInfo.filePath, code, 'utf-8');
      return NextResponse.json({
        success: true,
        path: pgInfo.filename,
        playgroundType: pgInfo.type,
        message: `Successfully forked to ${pgInfo.filename}!`,
      });
    }

    // 3. Save Workbook
    if (type === 'workbook') {
      const loc = getTopicLocation(chapterId, volumeId);
      if (!loc) {
        return NextResponse.json({ success: false, error: `Topic or Chapter ${chapterId} not found` }, { status: 404 });
      }
      const dir = path.join(loc.fullPath, target);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      const filePath = path.join(dir, `${tier}_workbook.${loc.ext}`);
      fs.writeFileSync(filePath, code, 'utf-8');
      return NextResponse.json({ success: true, path: filePath });
    }

    return NextResponse.json({ success: false, error: 'Invalid save type' }, { status: 400 });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
