'use client';

import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { RooflineCanvas } from './RooflineCanvas';
import { MemoryArrayCanvas } from './MemoryArrayCanvas';
import { PipelineFlowCanvas } from './PipelineFlowCanvas';
import {
  RooflineDiagramPayloadSchema,
  StridedAccessPayloadSchema,
  PipelineFlowPayloadSchema,
} from './types';
import { GenericDiagramRenderer } from '../TextbookDiagrams';

interface DynamicDiagramEngineProps {
  type: string;
  payload: string | object;
}

export function DynamicDiagramEngine({ type, payload }: DynamicDiagramEngineProps) {
  let parsedData: any = {};

  if (typeof payload === 'string') {
    const trimmed = payload.trim();
    if (trimmed.length > 0) {
      try {
        parsedData = JSON.parse(trimmed);
      } catch (err: any) {
        return (
          <div className="my-6 p-4 rounded-xl border border-rose-500/30 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>Invalid JSON payload in diagram:<strong>{type}</strong> — {err.message}</span>
          </div>
        );
      }
    }
  } else if (typeof payload === 'object' && payload !== null) {
    parsedData = payload;
  }

  const cleanType = type.toLowerCase().trim();

  // 1. Hardware Roofline Model
  if (cleanType.includes('roofline') || cleanType.includes('machine-balance') || cleanType.includes('arithmetic-intensity')) {
    const validated = RooflineDiagramPayloadSchema.safeParse(parsedData);
    const data = validated.success ? validated.data : parsedData;
    return <RooflineCanvas payload={data} />;
  }

  // 2. Strided Vector Access & Multi-Stream (SAXPY)
  if (cleanType.includes('strided') || cleanType.includes('saxpy') || cleanType.includes('dual-stream') || cleanType.includes('multistream')) {
    const validated = StridedAccessPayloadSchema.safeParse(parsedData);
    const data = validated.success ? validated.data : parsedData;
    return <MemoryArrayCanvas payload={data} />;
  }

  // 3. Pipeline Flow & Traffic Stages
  if (cleanType.includes('pipeline') || cleanType.includes('flow') || cleanType.includes('traffic') || cleanType.includes('fused')) {
    const validated = PipelineFlowPayloadSchema.safeParse(parsedData);
    const data = validated.success ? validated.data : parsedData;
    return <PipelineFlowCanvas payload={data} />;
  }

  // 4. Fallback to existing registered diagram primitives in TextbookDiagrams
  return (
    <div className="hardware-diagram-viewport rounded-2xl overflow-hidden">
      <GenericDiagramRenderer type={type} payload={typeof payload === 'string' ? payload : JSON.stringify(payload)} />
    </div>
  );
}

