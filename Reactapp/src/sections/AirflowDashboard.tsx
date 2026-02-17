import { useState } from 'react';
import { 
  Server, Play, CheckCircle, XCircle, Clock, 
  RotateCcw, ChevronDown, ChevronRight, FileText
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle 
} from '@/components/ui/dialog';

interface AirflowDashboardProps {
  data: {
    dagId: string;
    status: string;
    lastRun: string;
    nextRun: string;
    totalRuns: number;
    schedule: string;
    tasks: {
      id: string;
      name: string;
      status: string;
      startTime?: string;
      endTime?: string;
      duration?: number;
      logs?: string[];
    }[];
  };
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'success':
      return <CheckCircle className="w-5 h-5 text-green-400" />;
    case 'failed':
      return <XCircle className="w-5 h-5 text-red-400" />;
    case 'running':
      return <Clock className="w-5 h-5 text-blue-400 animate-pulse" />;
    default:
      return <Clock className="w-5 h-5 text-slate-400" />;
  }
};

const getStatusBadge = (status: string) => {
  const variants: Record<string, string> = {
    success: 'bg-green-500/20 text-green-400 border-green-500/30',
    failed: 'bg-red-500/20 text-red-400 border-red-500/30',
    running: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    pending: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    skipped: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
  };
  return variants[status] || 'bg-slate-500/20 text-slate-400 border-slate-500/30';
};

export default function AirflowDashboard({ data }: AirflowDashboardProps) {
  const [selectedTask, setSelectedTask] = useState<typeof data.tasks[0] | null>(null);
  const [expandedTasks, setExpandedTasks] = useState<Set<string>>(new Set());

  const toggleTask = (taskId: string) => {
    const newExpanded = new Set(expandedTasks);
    if (newExpanded.has(taskId)) {
      newExpanded.delete(taskId);
    } else {
      newExpanded.add(taskId);
    }
    setExpandedTasks(newExpanded);
  };

  const completedTasks = data.tasks.filter(t => t.status === 'success').length;
  const progress = (completedTasks / data.tasks.length) * 100;

  return (
    <div className="space-y-6">
      {/* DAG Header */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-500/20 rounded-lg">
                <Server className="w-6 h-6 text-green-400" />
              </div>
              <div>
                <CardTitle className="text-white">{data.dagId}</CardTitle>
                <CardDescription className="text-slate-400">
                  Schedule: {data.schedule}
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge 
                variant="outline" 
                className={data.status === 'running' 
                  ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                  : 'bg-slate-500/20 text-slate-400 border-slate-500/30'
                }
              >
                {data.status}
              </Badge>
              <Button variant="outline" size="sm">
                <Play className="w-4 h-4 mr-2" />
                Trigger
              </Button>
              <Button variant="outline" size="icon">
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <p className="text-sm text-slate-400 mb-1">Last Run</p>
              <p className="text-lg font-semibold text-white">
                {new Date(data.lastRun).toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <p className="text-sm text-slate-400 mb-1">Next Run</p>
              <p className="text-lg font-semibold text-white">
                {new Date(data.nextRun).toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <p className="text-sm text-slate-400 mb-1">Total Runs</p>
              <p className="text-lg font-semibold text-white">{data.totalRuns}</p>
            </div>
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <p className="text-sm text-slate-400 mb-1">Success Rate</p>
              <p className="text-lg font-semibold text-green-400">98.5%</p>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Pipeline Progress</span>
              <span className="text-white">{completedTasks}/{data.tasks.length} tasks</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        </CardContent>
      </Card>

      {/* Tasks List */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white text-lg">DAG Tasks</CardTitle>
          <CardDescription className="text-slate-400">
            Execution flow of the ETL pipeline
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {data.tasks.map((task, index) => (
              <div key={task.id} className="border border-slate-800 rounded-lg overflow-hidden">
                <div 
                  className="flex items-center justify-between p-4 bg-slate-950 hover:bg-slate-900 cursor-pointer transition-colors"
                  onClick={() => toggleTask(task.id)}
                >
                  <div className="flex items-center gap-4">
                    <span className="text-slate-500 font-mono text-sm">{index + 1}</span>
                    {getStatusIcon(task.status)}
                    <span className="font-medium text-white">{task.name}</span>
                    <Badge variant="outline" className={getStatusBadge(task.status)}>
                      {task.status}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4">
                    {task.duration && (
                      <span className="text-sm text-slate-400">
                        {task.duration}s
                      </span>
                    )}
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedTask(task);
                      }}
                    >
                      <FileText className="w-4 h-4 mr-2" />
                      Logs
                    </Button>
                    {expandedTasks.has(task.id) ? (
                      <ChevronDown className="w-5 h-5 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-slate-400" />
                    )}
                  </div>
                </div>
                
                {expandedTasks.has(task.id) && (
                  <div className="p-4 bg-slate-950 border-t border-slate-800">
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-slate-400">Start Time:</span>
                        <p className="text-white">
                          {task.startTime ? new Date(task.startTime).toLocaleString() : 'N/A'}
                        </p>
                      </div>
                      <div>
                        <span className="text-slate-400">End Time:</span>
                        <p className="text-white">
                          {task.endTime ? new Date(task.endTime).toLocaleString() : 'Running...'}
                        </p>
                      </div>
                      <div>
                        <span className="text-slate-400">Duration:</span>
                        <p className="text-white">{task.duration ? `${task.duration}s` : 'N/A'}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Task Logs Dialog */}
      <Dialog open={!!selectedTask} onOpenChange={() => setSelectedTask(null)}>
        <DialogContent className="max-w-3xl bg-slate-900 border-slate-800 text-white">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedTask && getStatusIcon(selectedTask.status)}
              Task Logs: {selectedTask?.name}
            </DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-[400px] w-full rounded-md border border-slate-800 p-4 bg-slate-950">
            <div className="font-mono text-sm space-y-2">
              <p className="text-green-400">[2024-01-15 10:00:00] INFO - Starting task: {selectedTask?.name}</p>
              <p className="text-slate-400">[2024-01-15 10:00:01] INFO - Loading configuration...</p>
              <p className="text-slate-400">[2024-01-15 10:00:02] INFO - Connecting to data source...</p>
              <p className="text-blue-400">[2024-01-15 10:00:03] INFO - Processing records...</p>
              <p className="text-slate-400">[2024-01-15 10:00:15] INFO - Processed 10,000 records</p>
              <p className="text-slate-400">[2024-01-15 10:00:28] INFO - Processed 20,000 records</p>
              <p className="text-slate-400">[2024-01-15 10:00:35] INFO - Validation passed</p>
              <p className="text-green-400">[2024-01-15 10:00:36] INFO - Task completed successfully</p>
              <p className="text-slate-500">[2024-01-15 10:00:36] INFO - Duration: {selectedTask?.duration}s</p>
            </div>
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </div>
  );
}
