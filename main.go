package main

import (
	"gopkg.in/mgo.v2/bson"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
	mgo "gopkg.in/mgo.v2"
)

//Definition properties
type Definition struct {
	Word    string   `json:"word"`
	Meaning string   `json:"m34eaning23"`
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
		Word:"hi",
		Meaning:"greeting",
		Usage:[]string{"when you meet someone"},
	}
	err = c.Insert(definition2)
	var definition1 = Definition{
		Word:"bye",
		Meaning:"greeting",
		Usage:[]string{"when you leave someone"},
	}
	err = c.Insert(definition1)

	var def []Definition
	c.Find(bson.M{"word":"bye"}).All(&def)

	fmt.Println(def)	
	// log.Fatal(run())
	// c.RemoveAll(nil)
	fmt.Println("Program completed")
}

func makeMuxRouter() http.Handler {
	muxRouter := mux.NewRouter()
	muxRouter.HandleFunc("/definition", getDefinition).Methods("GET")
	// muxRouter.HandleFunc("/request", handleWriteBlock).Methods("POST")
	return muxRouter
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

func setupDB() {
	// Setup connection to our postgresql database
	connString := `user=postgres
					password=1234
					host=localhost
					port=5432
					dbname=dictionaryDatabase
					sslmode=disable`
	db, err := sql.Open("postgres", connString)
	if err != nil {
		panic(err)
	}

	// Check whether we can access the database by pinging it
	err = db.Ping()
	if err != nil {
		panic(err)
	}

	// Place our opened database into a `dbstruct` and assign it to `store` variable.
	// The `store` variable implements a `Store` interface. The `store` variable was
	// declared globally in `store.go` file.
	store = &dbStore{db: db}
}

func getDefinition(w http.ResponseWriter, r *http.Request) {
	// Retrieve people from postgresql database using our `store` interface variable's
	// `func (*dbstore) GetPerson` pointer receiver method defined in `store.go` file
	personList, err := store.GetPerson()

	// Convert the `personList` variable to JSON
	personListBytes, err := json.Marshal(personList)
	if err != nil {
		fmt.Println(fmt.Errorf("Error: %v", err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	// Write JSON list of persons to response
	w.Write(personListBytes)
}
