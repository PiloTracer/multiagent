@echo off
echo Stopping all running Docker containers...

REM Stop all containers started by docker-compose
docker-compose down

REM Optional: If you only want to stop the containers without removing them, use:
REM docker-compose stop

echo Docker containers stopped successfully.
pause