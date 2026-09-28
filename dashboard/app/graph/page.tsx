"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import { getGraph, getResources, isLive } from "@/lib/api";
import type { GraphData } from "@/lib/types";

// ForceGraph requires the window object, so we disable SSR
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function GraphPage() {
  const [namespaces, setNamespaces] = useState<string[]>([]);
  const [selectedNs, setSelectedNs] = useState<string>("default");
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getResources().then((data) => {
      if (data && data.namespaces.length > 0) {
        setNamespaces(data.namespaces);
        if (!data.namespaces.includes(selectedNs)) {
          setSelectedNs(data.namespaces[0]);
        }
      }
    });
  }, []);

  useEffect(() => {
    setLoading(true);
    getGraph(selectedNs).then((data) => {
      if (data) setGraphData(data);
      setLoading(false);
    });
  }, [selectedNs]);

  return (
    <>
      <TopBar window={3600} onWindow={() => {}} auto={false} onAuto={() => {}} live={isLive()} />
      <div className="flex min-h-[calc(100vh-52px)]">
        <Sidebar />
        <main className="flex-1 min-w-0 p-6 flex flex-col h-[calc(100vh-52px)]">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Knowledge Graph Visualizer</h2>
            <div className="flex items-center gap-3">
              <span className="text-[13px] text-muted">Namespace:</span>
              <select 
                value={selectedNs} 
                onChange={(e) => setSelectedNs(e.target.value)}
                className="font-mono text-sm border border-line rounded-md px-3 py-1.5 bg-bg"
              >
                {namespaces.map(ns => (
                  <option key={ns} value={ns}>{ns}</option>
                ))}
                {namespaces.length === 0 && <option value="default">default</option>}
              </select>
            </div>
          </div>
          
          <div className="flex-1 bg-surface border border-line rounded-xl overflow-hidden relative">
            {loading && <div className="absolute inset-0 flex items-center justify-center bg-surface/50 z-10 text-muted">Loading graph...</div>}
            {graphData.nodes.length === 0 && !loading ? (
              <div className="absolute inset-0 flex items-center justify-center text-muted">
                No graph edges found in namespace '{selectedNs}'. Use graph_add_edge() to build your knowledge graph!
              </div>
            ) : (
              <ForceGraph2D
                graphData={graphData}
                nodeLabel="id"
                nodeColor={() => "#e8623b"}
                nodeRelSize={6}
                linkColor={() => "#ece6df"}
                linkDirectionalArrowLength={3.5}
                linkDirectionalArrowRelPos={1}
                linkLabel="label"
              />
            )}
          </div>
        </main>
      </div>
    </>
  );
}
