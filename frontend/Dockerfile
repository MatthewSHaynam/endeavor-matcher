FROM nginx:alpine

# Copy the frontend code into the Nginx HTML folder.
COPY public/ /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
