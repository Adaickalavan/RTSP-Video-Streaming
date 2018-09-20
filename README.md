# DictionaryAPI

This project builds a REST web API to POST and GET word defintions to/from a dictionary stored in a MongoDB database. Dockerfile and Docker-compose files are provided to containerize the deployment of Go code and MongoDB database. 

Docker commands:

1. Build image of Go code: 
   + `docker build -t "dictionaryapi" .`
2. Create and run all containers: 
   + `docker-compose up`
3. Tear down all containers and stored volume: 
   + `docker-compose down -v`

Further extension of functionality and more description of the code will be proivded later.
