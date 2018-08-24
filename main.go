package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"gopkg.in/mgo.v2/bson"

	"github.com/joho/godotenv"
	mgo "gopkg.in/mgo.v2"
)

//Hooks that may be overridden for testing
var inputReader io.Reader = os.Stdin
var outputWriter io.Writer = os.Stdout

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

	var db *mgo.Database
	const COLLECTION = "words"

	//Create a new document
	c := session.DB("dictionary").C("words")

	//Insert documents into MongoDB
	var word2 = Word{
		Value:   "hi",
		Meaning: "greeting",
		Usage:   []string{"when you meet someone"},
	}
	err = c.Insert(word2)
	var word1 = Word{
		Value:   "bye",
		Meaning: "greeting",
		Usage:   []string{"when you leave someone"},
	}
	err = c.Insert(word1)

	//Update document in MongoDB
	fmt.Println("Updating doc")
	var def Word
	// c.Find(bson.M{"word": "bye"}).One(&def)
	err = c.Update(bson.M{"word": "bye"}, &Word{Value: "dfrebye", Meaning: "greeting", Usage: []string{"chandwdged"}})
	checkError(err)

	//Fetch documents from MongoDB
	var defs []Word
	c.Find(bson.M{"meaning": "greeting"}).All(&defs)

	fmt.Println(defs)
	fmt.Println(def)
	fmt.Println("Program completed")

	run()

}

func run() error {
	mux := makeMuxRouter()
	httpAddr := os.Getenv("ADDR")
	log.Println("Listening on ", httpAddr)
	s := &http.Server{
		Addr:           ":" + httpAddr,
		Handler:        mux,
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}
	if err := s.ListenAndServe(); err != nil {
		return err
	}
	return nil
}

func checkError(err error) bool {
	if err != nil {
		fmt.Fprintln(outputWriter, err.Error())
		return true
	}
	return false
}
