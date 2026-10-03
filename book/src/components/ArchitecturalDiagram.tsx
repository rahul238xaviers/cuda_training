'use client';

import React from 'react';
import { GenericDiagramRenderer } from './TextbookDiagrams';

interface ArchitecturalDiagramProps {
  type: string;
  payload?: string;
  title?: string;
  subtitle?: string;
}

/**
 * Pure presentation wrapper delegating to the generic vector diagram renderer.
 * Zero hardcoded chapter IDs or static domain text.
 */
export function ArchitecturalDiagram({ type, payload = '', title, subtitle }: ArchitecturalDiagramProps) {
  const effectivePayload = payload || JSON.stringify({ title, subtitle });
  return <GenericDiagramRenderer type={type} payload={effectivePayload} />;
}
