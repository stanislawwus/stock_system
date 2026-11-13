"""FastAPI layer, prefix /api/v1. Routers are thin: validate -> call service -> return schema. Long
operations return job_id.
"""
