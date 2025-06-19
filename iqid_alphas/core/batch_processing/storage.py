"""
Batch Processing Storage and Results Management

Handles storage, retrieval, and management of batch processing results
with support for different storage backends and data formats.
"""

import json
import pickle
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import datetime
import pandas as pd

from .types import BatchJob, BatchResult, ProcessingStatus, WorkerStats, ResourceMetrics

logger = logging.getLogger(__name__)


class ResultsStorage:
    """
    Storage backend for batch processing results.
    
    Supports multiple storage formats and provides efficient
    querying and retrieval of processing results.
    """
    
    def __init__(self, 
                 storage_path: Union[str, Path],
                 storage_format: str = "sqlite",
                 compress_results: bool = True):
        """
        Initialize results storage.
        
        Parameters
        ----------
        storage_path : Union[str, Path]
            Path to storage directory or file
        storage_format : str
            Storage format ('sqlite', 'json', 'pickle', 'parquet')
        compress_results : bool
            Whether to compress stored results
        """
        self.storage_path = Path(storage_path)
        self.storage_format = storage_format.lower()
        self.compress_results = compress_results
        
        # Create storage directory
        if self.storage_format != "sqlite":
            self.storage_path.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backend
        self._initialize_storage()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_storage(self):
        """Initialize storage backend."""
        if self.storage_format == "sqlite":
            self._initialize_sqlite()
        elif self.storage_format in ["json", "pickle", "parquet"]:
            # File-based storage, no initialization needed
            pass
        else:
            raise ValueError(f"Unsupported storage format: {self.storage_format}")
    
    def _initialize_sqlite(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_jobs (
                job_id TEXT PRIMARY KEY,
                input_path TEXT,
                output_path TEXT,
                pipeline_stage TEXT,
                status TEXT,
                success BOOLEAN,
                started_at REAL,
                completed_at REAL,
                processing_time_seconds REAL,
                worker_id TEXT,
                error_message TEXT,
                results TEXT,
                metadata TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_results (
                result_id TEXT PRIMARY KEY,
                total_jobs INTEGER,
                completed_jobs INTEGER,
                failed_jobs INTEGER,
                success_rate REAL,
                total_processing_time REAL,
                average_processing_time REAL,
                throughput_jobs_per_hour REAL,
                started_at REAL,
                completed_at REAL,
                metadata TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS worker_stats (
                worker_id TEXT,
                timestamp REAL,
                jobs_processed INTEGER,
                total_processing_time REAL,
                average_processing_time REAL,
                errors_encountered INTEGER,
                throughput REAL,
                status TEXT,
                PRIMARY KEY (worker_id, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_metrics (
                timestamp REAL PRIMARY KEY,
                cpu_percent REAL,
                memory_used_gb REAL,
                memory_available_gb REAL,
                disk_used_gb REAL,
                disk_available_gb REAL,
                active_workers INTEGER
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON batch_jobs(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_timestamp ON batch_jobs(started_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_worker_stats_timestamp ON worker_stats(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resource_metrics_timestamp ON resource_metrics(timestamp)')
        
        conn.commit()
        conn.close()
        
        # Use module logger since self.logger might not be initialized
        logger.info(f"SQLite storage initialized at {self.storage_path}")
    
    def store_job(self, job: BatchJob):
        """Store a batch job result."""
        try:
            if self.storage_format == "sqlite":
                self._store_job_sqlite(job)
            elif self.storage_format == "json":
                self._store_job_json(job)
            elif self.storage_format == "pickle":
                self._store_job_pickle(job)
            else:
                raise ValueError(f"Unsupported storage format: {self.storage_format}")
            
            self.logger.debug(f"Stored job {job.job_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store job {job.job_id}: {e}")
    
    def _store_job_sqlite(self, job: BatchJob):
        """Store job in SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO batch_jobs (
                job_id, input_path, output_path, pipeline_stage, status, success,
                started_at, completed_at, processing_time_seconds, worker_id,
                error_message, results, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.job_id,
            str(job.input_path),
            str(job.output_path),
            job.pipeline_stage.value if job.pipeline_stage else None,
            job.status.value if job.status else None,
            job.success,
            job.started_at,
            job.completed_at,
            job.processing_time_seconds,
            job.worker_id,
            job.error_message,
            json.dumps(job.results) if job.results else None,
            json.dumps(job.metadata) if job.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def _store_job_json(self, job: BatchJob):
        """Store job as JSON file."""
        job_file = self.storage_path / f"job_{job.job_id}.json"
        
        # Convert job to serializable format
        job_data = asdict(job)
        job_data['status'] = job.status.value if job.status else None
        job_data['pipeline_stage'] = job.pipeline_stage.value if job.pipeline_stage else None
        
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2, default=str)
    
    def _store_job_pickle(self, job: BatchJob):
        """Store job as pickle file."""
        job_file = self.storage_path / f"job_{job.job_id}.pkl"
        
        with open(job_file, 'wb') as f:
            pickle.dump(job, f)
    
    def store_batch_result(self, result: BatchResult):
        """Store batch processing result."""
        try:
            if self.storage_format == "sqlite":
                self._store_batch_result_sqlite(result)
            elif self.storage_format == "json":
                self._store_batch_result_json(result)
            elif self.storage_format == "pickle":
                self._store_batch_result_pickle(result)
            
            self.logger.info("Stored batch result")
            
        except Exception as e:
            self.logger.error(f"Failed to store batch result: {e}")
    
    def _store_batch_result_sqlite(self, result: BatchResult):
        """Store batch result in SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        result_id = f"batch_{int(result.started_at)}"
        
        cursor.execute('''
            INSERT OR REPLACE INTO batch_results (
                result_id, total_jobs, completed_jobs, failed_jobs, success_rate,
                total_processing_time, average_processing_time, throughput_jobs_per_hour,
                started_at, completed_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id,
            result.total_jobs,
            result.completed_jobs,
            result.failed_jobs,
            result.success_rate,
            result.total_processing_time,
            result.average_processing_time,
            result.throughput_jobs_per_hour,
            result.started_at,
            result.completed_at,
            json.dumps(result.metadata) if result.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def _store_batch_result_json(self, result: BatchResult):
        """Store batch result as JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.storage_path / f"batch_result_{timestamp}.json"
        
        result_data = asdict(result)
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
    
    def _store_batch_result_pickle(self, result: BatchResult):
        """Store batch result as pickle file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.storage_path / f"batch_result_{timestamp}.pkl"
        
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
    
    def store_worker_stats(self, stats: WorkerStats):
        """Store worker statistics."""
        try:
            if self.storage_format == "sqlite":
                self._store_worker_stats_sqlite(stats)
            
        except Exception as e:
            self.logger.error(f"Failed to store worker stats: {e}")
    
    def _store_worker_stats_sqlite(self, stats: WorkerStats):
        """Store worker stats in SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        timestamp = datetime.datetime.now().timestamp()
        
        cursor.execute('''
            INSERT OR REPLACE INTO worker_stats (
                worker_id, timestamp, jobs_processed, total_processing_time,
                average_processing_time, errors_encountered, throughput, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stats.worker_id,
            timestamp,
            stats.jobs_processed,
            stats.total_processing_time,
            stats.average_processing_time,
            stats.errors_encountered,
            stats.throughput,
            stats.status
        ))
        
        conn.commit()
        conn.close()
    
    def store_resource_metrics(self, metrics: ResourceMetrics):
        """Store resource metrics."""
        try:
            if self.storage_format == "sqlite":
                self._store_resource_metrics_sqlite(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to store resource metrics: {e}")
    
    def _store_resource_metrics_sqlite(self, metrics: ResourceMetrics):
        """Store resource metrics in SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO resource_metrics (
                timestamp, cpu_percent, memory_used_gb, memory_available_gb,
                disk_used_gb, disk_available_gb, active_workers
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp,
            metrics.cpu_percent,
            metrics.memory_used_gb,
            metrics.memory_available_gb,
            metrics.disk_used_gb,
            metrics.disk_available_gb,
            metrics.active_workers
        ))
        
        conn.commit()
        conn.close()
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Retrieve a specific job by ID."""
        try:
            if self.storage_format == "sqlite":
                return self._get_job_sqlite(job_id)
            elif self.storage_format == "json":
                return self._get_job_json(job_id)
            elif self.storage_format == "pickle":
                return self._get_job_pickle(job_id)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve job {job_id}: {e}")
            return None
    
    def _get_job_sqlite(self, job_id: str) -> Optional[BatchJob]:
        """Get job from SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM batch_jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Convert row to BatchJob
        # This would need proper deserialization
        return None  # Simplified for brevity
    
    def query_jobs(self, 
                  status: Optional[ProcessingStatus] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query jobs with filters."""
        try:
            if self.storage_format == "sqlite":
                return self._query_jobs_sqlite(status, start_time, end_time, limit)
            else:
                return []  # File-based storage would need different implementation
                
        except Exception as e:
            self.logger.error(f"Failed to query jobs: {e}")
            return []
    
    def _query_jobs_sqlite(self, 
                          status: Optional[ProcessingStatus],
                          start_time: Optional[float],
                          end_time: Optional[float],
                          limit: Optional[int]) -> List[Dict[str, Any]]:
        """Query jobs from SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM batch_jobs WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if start_time:
            query += " AND started_at >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND started_at <= ?"
            params.append(end_time)
        
        query += " ORDER BY started_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to dictionaries
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            if self.storage_format == "sqlite":
                return self._get_statistics_sqlite()
            else:
                return self._get_statistics_file()
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def _get_statistics_sqlite(self) -> Dict[str, Any]:
        """Get statistics from SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Job statistics
        cursor.execute('SELECT COUNT(*) FROM batch_jobs')
        total_jobs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM batch_jobs WHERE status = "completed"')
        completed_jobs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM batch_jobs WHERE status = "failed"')
        failed_jobs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM batch_results')
        total_batches = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'total_batches': total_batches,
            'success_rate': (completed_jobs / max(total_jobs, 1)) * 100
        }
    
    def _get_statistics_file(self) -> Dict[str, Any]:
        """Get statistics for file-based storage."""
        if self.storage_format == "json":
            job_files = list(self.storage_path.glob("job_*.json"))
            batch_files = list(self.storage_path.glob("batch_result_*.json"))
        elif self.storage_format == "pickle":
            job_files = list(self.storage_path.glob("job_*.pkl"))
            batch_files = list(self.storage_path.glob("batch_result_*.pkl"))
        else:
            job_files = []
            batch_files = []
        
        return {
            'total_jobs': len(job_files),
            'total_batches': len(batch_files),
            'storage_format': self.storage_format
        }
    
    def export_to_csv(self, output_path: Union[str, Path]):
        """Export job data to CSV format."""
        output_path = Path(output_path)
        
        try:
            if self.storage_format == "sqlite":
                # Export SQLite data to CSV
                conn = sqlite3.connect(self.storage_path)
                
                # Export jobs
                jobs_df = pd.read_sql_query("SELECT * FROM batch_jobs", conn)
                jobs_df.to_csv(output_path.parent / f"{output_path.stem}_jobs.csv", index=False)
                
                # Export batch results
                results_df = pd.read_sql_query("SELECT * FROM batch_results", conn)
                results_df.to_csv(output_path.parent / f"{output_path.stem}_results.csv", index=False)
                
                conn.close()
                
                self.logger.info(f"Data exported to CSV files at {output_path.parent}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to CSV: {e}")
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old records to save space."""
        try:
            cutoff_time = datetime.datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            if self.storage_format == "sqlite":
                conn = sqlite3.connect(self.storage_path)
                cursor = conn.cursor()
                
                # Delete old jobs
                cursor.execute('DELETE FROM batch_jobs WHERE started_at < ?', (cutoff_time,))
                deleted_jobs = cursor.rowcount
                
                # Delete old resource metrics
                cursor.execute('DELETE FROM resource_metrics WHERE timestamp < ?', (cutoff_time,))
                deleted_metrics = cursor.rowcount
                
                # Delete old worker stats
                cursor.execute('DELETE FROM worker_stats WHERE timestamp < ?', (cutoff_time,))
                deleted_stats = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Cleaned up {deleted_jobs} jobs, {deleted_metrics} metrics, "
                               f"{deleted_stats} worker stats older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
