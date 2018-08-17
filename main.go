package main

import (
	"fmt"
	"io"
	"log"
	"os"

	"gopkg.in/mgo.v2/bson"

	"github.com/joho/godotenv"
	mgo "gopkg.in/mgo.v2"
)

var inputReader io.Reader = os.Stdin
var outputWriter io.Writer = os.Stdout

//Definition properties
type Definition struct {
	Word    string   `json:"word"`
	Meaning string   `json:"meaning"`
	Usage   []string `json:"usage"`
}

func main() {
	//Load .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatal(err)
	}

	//Create a session
	session, err := mgo.Dial("localhost")
	if err != nil {
		panic(err)
	}
	defer session.Close()
	//Create additional sessions
	anotherSession := session.Copy()
	defer anotherSession.Close()

	//Create a new document
	c := session.DB("dictionary").C("definitions")
	var definition2 = Definition{
		Word:    "hi",
		Meaning: "greeting",
		Usage:   []string{"when you meet someone"},
	}
	err = c.Insert(definition2)
	var definition1 = Definition{
		Word:    "bye",
		Meaning: "greeting",
		Usage:   []string{"when you leave someone"},
	}
	err = c.Insert(definition1)

	index := mgo.Index{
		Key:        []string{"word"}, //Index key fields; prefix name with (-) dash for descending order
		Unique:     true,             //Prevent two documents from having the same key
		DropDups:   true,             //Drop documents with same index
		Background: true,             //Build index in background and return immediately
		Sparse:     true,             //Only index documents containing the Key fields
	}
	err = c.EnsureIndex(index)
	checkError(err)

	var def []Definition
	c.Find(bson.M{"word": "bye"}).All(&def)

	fmt.Println(def)
	// log.Fatal(run())
	// c.RemoveAll(nil)
	fmt.Println("Program completed")
}

// func run() error {
// 	mux := makeMuxRouter()
// 	httpAddr := os.Getenv("ADDR")
// 	log.Println("Listening on ", httpAddr)
// 	s := &http.Server{
// 		Addr:           ":" + httpAddr,
// 		Handler:        mux,
// 		ReadTimeout:    10 * time.Second,
// 		WriteTimeout:   10 * time.Second,
// 		MaxHeaderBytes: 1 << 20,
// 	}
// 	if err := s.ListenAndServe(); err != nil {
// 		return err
// 	}
// 	return nil
// }

func checkError(err error) bool {
	if err != nil {
		fmt.Fprintln(outputWriter, err.Error())
		return true
	}
	return false
}
