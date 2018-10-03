#Start from a Debian image with the latest version of GO 
#installed and a workspace (GOPATH) configure at /go.
FROM golang:onbuild

LABEL Author Adaickalavan Meiyappan

# Document port which the service listens on
EXPOSE 8080
