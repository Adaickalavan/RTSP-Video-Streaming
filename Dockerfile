#Start from a Debian image with the latest version of GO 
#installed and a workspace (GOPATH) configure at /go.
# FROM golang:onbuild
FROM golang:onbuild

# #Copy the local package files to the container's workspace 
# ADD . /go/src/github.com/adaickalavan/DictionaryAPI

# #Build the DictionaryAPI command inside the container
# RUN go install github.com/adaickalavan/DictionaryAPI

# #Run the DictionaryAPI command by default when container starts
# ENTRYPOINT /go/bin/DictionaryAPI

#The service listens on port
EXPOSE 8080

