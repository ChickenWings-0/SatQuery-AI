# SatQuery AI — the frontend, built and served by nginx.
#
# `/v1` is proxied to the API service, which is what makes the deployed page
# same-origin — the same shape the Vite dev proxy gives development, so SSE and
# multipart behave identically. It is also why the API's CORS list must allow
# DELETE: the proxy does not hide a missing method the way same-origin does.
#
#   docker build -t satquery-web:latest frontend

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json .nvmrc ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
