package main

import (
	"document"
	"fmt"
	"log"

	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

// func setupDB() {
// 	// Setup connection to our postgresql database
// 	connString := `user=postgres
// 					password=1234
// 					host=localhost
// 					port=5432
// 					dbname=dictionaryDatabase
// 					sslmode=disable`
// 	db, err := sql.Open("postgres", connString)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// Check whether we can access the database by pinging it
// 	err = db.Ping()
// 	if err != nil {
// 		panic(err)
// 	}

// 	// Place our opened database into a `dbstruct` and assign it to `store` variable.
// 	// The `store` variable implements a `Store` interface. The `store` variable was
// 	// declared globally in `store.go` file.
// 	store = &dbStore{db: db}
// }

//Dictionary contains server and database strings
type Dictionary struct {
	Server   string
	Database string
}

var db *mgo.Database

//COLLECTION is the name of collection within Dictionary database
const COLLECTION = "words"

//Connect connects to the database
func (dictionary Dictionary) Connect() {
	session, err := mgo.Dial(dictionary.Server)
	if err != nil {
		log.Fatal(err)
	}
}

func (dictionary *Dictionary) ensureIndex(c *mgo.Collection) {
	//Ensure index in MongoDB
	index := mgo.Index{
		Key:        []string{"value"}, //Index key fields; prefix name with (-) dash for descending order
		Unique:     true,              //Prevent two documents from having the same key
		DropDups:   true,              //Drop documents with same index
		Background: true,              //Build index in background and return immediately
		Sparse:     true,              //Only index documents containing the Key fields
	}
	err := c.EnsureIndex(index)
	checkError(err)
}

//FindAll retrieves the all Word by its Value from dictionary
func (dictionary *Dictionary) FindAll(value string) ([]document.Word, error) {
	var words []document.Word
	err := db.C(COLLECTION).Find(bson.M{}).One(&words)
	return words, err
}

//FindByValue retrieves the Word by its Value from dictionary
func (dictionary *Dictionary) FindByValue(value string) (document.Word, error) {
	var word document.Word
	err := db.C(COLLECTION).Find(bson.M{"value": value}).One(&word)
	return word, err
}

//FindByValue retrieves the Word by its Value from dictionary
func (dictionary *Dictionary) Insert(word Word) (document.Word, error) {
	var word document.Word
	err := db.C(COLLECTION).Find(bson.M{}).One(&word)
	return word, err
}

func checkError(err error) bool {
	if err != nil {
		fmt.Println(err.Error())
		return true
	}
	return false
}
