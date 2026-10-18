import { z } from 'zod';

// Workload spec for Roofline
export const RooflineWorkloadSchema = z.object({
  id: z.string(),
  name: z.string(),
  intensity: z.number(),
  tag: z.string().optional(),
  expr: z.string().optional(),
  busTraffic: z.string().optional(),
  explanation: z.string().optional(),
  badgeColor: z.enum(['sky', 'amber', 'emerald', 'purple', 'rose']).optional(),
  isMemoryBound: z.boolean().optional(),
  diagnosis: z.string().optional(),
});

export type RooflineWorkload = z.infer<typeof RooflineWorkloadSchema>;

// Architecture spec for Roofline
export const RooflineArchSchema = z.object({
  id: z.string(),
  name: z.string(),
  chip: z.string().optional(),
  tech: z.string().optional(),
  peakComputeTflops: z.number(),
  bandwidthGBs: z.number(),
  ridgePoint: z.number(),
  peakComputeStr: z.string(),
  bandwidthStr: z.string(),
  color: z.string().optional(),
});

export type RooflineArch = z.infer<typeof RooflineArchSchema>;

// Roofline diagram JSON payload schema
export const RooflineDiagramPayloadSchema = z.object({
  title: z.string().default('Hardware Roofline Model: Memory Bound vs Compute Bound'),
  subtitle: z.string().default('Interactive plot mapping arithmetic intensity against physical hardware ceilings.'),
  defaultArch: z.string().default('rtx5090'),
  archs: z.record(z.string(), RooflineArchSchema).optional(),
  workloads: z.array(RooflineWorkloadSchema).optional(),
});

export type RooflineDiagramPayload = z.infer<typeof RooflineDiagramPayloadSchema>;

// Strided Access / Multi-Stream diagram payload schema
export const StridedAccessPayloadSchema = z.object({
  title: z.string().default('Dual-Stream Vector Traversal: SAXPY'),
  subtitle: z.string().default('Two independent pointer streams traverse separate arrays simultaneously over the memory bus.'),
  numCells: z.number().default(6),
  cellW: z.number().default(64),
  streamX: z.object({
    name: z.string().default('x[ ]'),
    baseAddr: z.string().default('0x1000'),
    mode: z.string().default('READ'),
    ptrLabel: z.string().default('x_ptr'),
    cellLabel: z.string().default('LOAD'),
  }).optional(),
  streamY: z.object({
    name: z.string().default('y[ ]'),
    baseAddr: z.string().default('0x4000'),
    mode: z.string().default('READ+WRITE'),
    ptrLabel: z.string().default('y_ptr'),
    cellLabel: z.string().default('LD+ST'),
  }).optional(),
  fmaBox: z.object({
    equation: z.string().default('y[i] = α · x[i] + y[i]'),
    metrics: z.string().default('2 FLOPs / 12 bytes = 0.17 FLOPs/B'),
    regime: z.string().default('Memory Bandwidth Bound'),
  }).optional(),
  metrics: z.array(z.object({
    title: z.string(),
    value: z.string(),
    subtext: z.string().optional(),
    color: z.enum(['sky', 'amber', 'rose', 'emerald', 'purple']).default('sky'),
  })).optional(),
});

export type StridedAccessPayload = z.infer<typeof StridedAccessPayloadSchema>;

// Pipeline Flow / Execution Stage diagram payload schema
export const PipelineStageSchema = z.object({
  id: z.string(),
  name: z.string(),
  subtitle: z.string().optional(),
  color: z.enum(['sky', 'amber', 'purple', 'emerald', 'rose', 'cyan']).default('sky'),
  traffic: z.string().optional(),
  unit: z.string().optional(),
  latency: z.string().optional(),
  details: z.array(z.string()).optional(),
});

export type PipelineStage = z.infer<typeof PipelineStageSchema>;

export const PipelineFlowPayloadSchema = z.object({
  title: z.string().default('Hardware Execution Pipeline & Traffic Flow'),
  subtitle: z.string().optional(),
  stages: z.array(PipelineStageSchema).default([]),
  metrics: z.array(z.object({
    label: z.string(),
    value: z.string(),
    subtext: z.string().optional(),
    color: z.enum(['sky', 'amber', 'rose', 'emerald', 'purple', 'cyan']).default('sky'),
  })).optional(),
  comparison: z.object({
    unfused: z.object({
      title: z.string(),
      traffic: z.string(),
      description: z.string().optional(),
    }),
    fused: z.object({
      title: z.string(),
      traffic: z.string(),
      description: z.string().optional(),
    }),
  }).optional(),
});

export type PipelineFlowPayload = z.infer<typeof PipelineFlowPayloadSchema>;

// Generic diagram fallback
export const GenericDiagramPayloadSchema = z.record(z.string(), z.any());


