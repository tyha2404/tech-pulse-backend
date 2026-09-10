module.exports = {
  apps: [
    {
      name: "techpulse-backend",
      script: ".venv/bin/python",
      args: "-m uvicorn app.main:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      restart_delay: 3000,
      max_restarts: 10,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
