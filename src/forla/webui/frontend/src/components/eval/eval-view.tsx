/**
 * EvalView — top-level evaluation page with Datasets / Targets / Runs tabs.
 */

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DatasetPanel } from "@/components/eval/dataset-panel";
import { TargetPanel } from "@/components/eval/target-panel";
import { EvalRunsPanel } from "@/components/eval/eval-runs-panel";

export function EvalView() {
  return (
    <div className="flex flex-col h-full">
      <Tabs defaultValue="runs" className="flex flex-col h-full">
        <div className="px-3 pt-3">
          <TabsList className="h-8">
            <TabsTrigger value="runs" className="text-xs px-3">
              Runs
            </TabsTrigger>
            <TabsTrigger value="datasets" className="text-xs px-3">
              Datasets
            </TabsTrigger>
            <TabsTrigger value="targets" className="text-xs px-3">
              Targets
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="runs" className="flex-1 overflow-hidden mt-0">
          <EvalRunsPanel />
        </TabsContent>

        <TabsContent value="datasets" className="flex-1 overflow-hidden mt-0">
          <DatasetPanel />
        </TabsContent>

        <TabsContent value="targets" className="flex-1 overflow-hidden mt-0">
          <TargetPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
