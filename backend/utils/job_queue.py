import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

class JobQueue:
    """Simple in-memory job queue for managing generation tasks"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def add_job(self, job_id: str, job_data: Dict[str, Any]) -> str:
        """Add a new job to the queue"""
        with self.lock:
            self.jobs[job_id] = {
                **job_data,
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs"""
        with self.lock:
            return list(self.jobs.values())
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job with new data"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
                self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
                return True
            return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "cancelled"
                self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
                return True
            return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        with self.lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                return True
            return False
    
    def clear_all_jobs(self) -> int:
        """Clear all jobs from the queue"""
        with self.lock:
            count = len(self.jobs)
            self.jobs.clear()
            return count
    
    def get_jobs_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get jobs by status"""
        with self.lock:
            return [job for job in self.jobs.values() if job["status"] == status]
    
    def get_queued_jobs(self) -> List[Dict[str, Any]]:
        """Get all queued jobs"""
        return self.get_jobs_by_status("queued")
    
    def get_processing_jobs(self) -> List[Dict[str, Any]]:
        """Get all processing jobs"""
        return self.get_jobs_by_status("processing")
    
    def get_completed_jobs(self) -> List[Dict[str, Any]]:
        """Get all completed jobs"""
        return self.get_jobs_by_status("completed")
    
    def get_failed_jobs(self) -> List[Dict[str, Any]]:
        """Get all failed jobs"""
        return self.get_jobs_by_status("failed")
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed/failed jobs"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self.lock:
            jobs_to_delete = []
            for job_id, job in self.jobs.items():
                if job["status"] in ["completed", "failed", "cancelled"]:
                    job_time = datetime.fromisoformat(job["created_at"]).timestamp()
                    if job_time < cutoff_time:
                        jobs_to_delete.append(job_id)
            
            for job_id in jobs_to_delete:
                del self.jobs[job_id]
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self.lock:
            stats = {
                "total": len(self.jobs),
                "queued": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0
            }
            
            for job in self.jobs.values():
                status = job["status"]
                if status in stats:
                    stats[status] += 1
            
            return stats
