package database

import (
	"document"
	"fmt"
	"log"
	"os"
	"time"

	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

var db *mgo.Database

//COLLECTION is the name of collection within Dictionary database
const COLLECTION = "words"

//Dictionary contains server and database strings
type Dictionary struct {
	Server       string
	DatabaseName string
	Session      *mgo.Session
}

//Connect connects to the database
func (dictionary Dictionary) Connect() *mgo.Session {
	info := &mgo.DialInfo{
		Addrs:    []string{os.Getenv("SERVER")},
		Timeout:  60 * time.Second,
		Database: os.Getenv("DATABASENAME"),
	}
	session, err := mgo.DialWithInfo(info)
	if err != nil {
		log.Fatal(err)
	}
	db = session.DB(dictionary.DatabaseName)
	return session
}

//EnsureIndex creates an index field in the collection
func (dictionary *Dictionary) EnsureIndex(fields []string) {
	//Ensure index in MongoDB
	index := mgo.Index{
		Key:        fields, //Index key fields; prefix name with (-) dash for descending order
		Unique:     true,   //Prevent two documents from having the same key
		DropDups:   true,   //Drop documents with same index
		Background: true,   //Build index in background and return immediately
		Sparse:     true,   //Only index documents containing the Key fields
	}
	err := db.C(COLLECTION).EnsureIndex(index)
	checkError(err)
}

//FindAll retrieves all Word by its Value from dictionary
func (dictionary *Dictionary) FindAll() ([]document.Word, error) {
	var words []document.Word
	err := db.C(COLLECTION).Find(bson.M{}).All(&words)
	return words, err
}

//FindByValue retrieves the Word by its Value from dictionary
func (dictionary *Dictionary) FindByValue(value string) (document.Word, error) {
	var word document.Word
	err := db.C(COLLECTION).Find(bson.M{"value": value}).One(&word)
	return word, err
}

//Insert inserts the Word into the dictionary
func (dictionary *Dictionary) Insert(word document.Word) error {
	err := db.C(COLLECTION).Insert(&word)
	return err
}

//Delete deletes the Word from dictionary
func (dictionary *Dictionary) Delete(word document.Word) error {
	err := db.C(COLLECTION).Remove(&word)
	return err
}

func checkError(err error) bool {
	if err != nil {
		fmt.Println(err.Error())
		return true
	}
	return false
}
