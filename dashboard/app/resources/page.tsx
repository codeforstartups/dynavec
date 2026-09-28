"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import { getResources, isLive } from "@/lib/api";
import type { ResourcesInfo } from "@/lib/types";

export default function ResourcesPage() {
  const [resources, setResources] = useState<ResourcesInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getResources().then((data) => {
      setResources(data);
      setLoading(false);
    });
  }, []);

  return (
    <>
      <TopBar window={3600} onWindow={() => {}} auto={false} onAuto={() => {}} live={isLive()} />
      <div className="flex min-h-[calc(100vh-52px)]">
        <Sidebar />
        <main className="flex-1 min-w-0 p-6">
          <h2 className="text-xl font-bold mb-4">Buckets &amp; Indexes</h2>
          
          {loading ? (
            <div className="text-muted">Loading resources...</div>
          ) : resources ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-surface border border-line rounded-xl p-5">
                <h3 className="font-semibold text-[15px] mb-4">DynamoDB Table</h3>
                <div className="space-y-3 text-[13.5px]">
                  <div className="flex justify-between">
                    <span className="text-muted">Name</span>
                    <span className="font-mono">{resources.table}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Status</span>
                    <span className={`font-mono ${resources.table_status === 'ACTIVE' ? 'text-ok' : 'text-accent'}`}>{resources.table_status}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Item Count</span>
                    <span className="font-mono">{resources.table_item_count.toLocaleString()}</span>
                  </div>
                </div>
              </div>

              <div className="bg-surface border border-line rounded-xl p-5">
                <h3 className="font-semibold text-[15px] mb-4">S3 Vectors</h3>
                <div className="space-y-3 text-[13.5px]">
                  <div className="flex justify-between">
                    <span className="text-muted">Bucket</span>
                    <span className="font-mono">{resources.vector_bucket}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Index Profile</span>
                    <span className="font-mono">{resources.index}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Vector Dimension</span>
                    <span className="font-mono">{resources.dimension}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Distance Metric</span>
                    <span className="font-mono">{resources.distance_metric}</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-surface border border-line rounded-xl p-5 md:col-span-2">
                <h3 className="font-semibold text-[15px] mb-4">Namespaces ({resources.namespaces.length})</h3>
                <div className="flex flex-wrap gap-2">
                  {resources.namespaces.map((ns) => (
                    <div key={ns} className="px-3 py-1.5 border border-line rounded-md text-[13px] bg-bg font-mono">
                      {ns}
                    </div>
                  ))}
                  {resources.namespaces.length === 0 && <div className="text-muted italic">No namespaces found</div>}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-err">Failed to load resources. Is dynavec.dashboard running?</div>
          )}
        </main>
      </div>
    </>
  );
}
