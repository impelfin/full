# step 1 : Base Images
FROM node:18.16.0

# step 2 : Package Install
RUN apt -y update && apt -y upgrade && apt -y install git net-tools vim

# step 3 : Specify a working directory
WORKDIR '/root'

# step 4 : Config file copy
COPY weather.tar.gz .

# step 5 : install express
RUN tar xvzf weather.tar.gz
WORKDIR '/root/weather'
RUN npm install
RUN npm install -g nodemon

# step 6 : open port
EXPOSE 8000

# step 7 : Execution Program
CMD ["nodemon", "app.js"]