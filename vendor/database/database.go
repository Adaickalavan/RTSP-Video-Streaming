package main

import (
	"database/sql"

	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)



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

func (word Words) ensureIndex(c *Collection) {
	//Ensure index in MongoDB
	ind)ex := mgo.Index{
		Key:        []string{"value"}, //Index key fields; prefix name with (-) dash for descending order
		Unique:     true,              //Prevent two documents from having the same key
		DropDups:   true,              //Drop documents with same index
		Background: true,              //Build index in background and return immediately
		Sparse:     true,              //Only index documents containing the Key fields
	}
	err = c.EnsureIndex(index)
	checkError(err)
}
