package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
)

func main() {
	//Load .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatal(err)
	}

	run()
	fmt.Println("yess")
}

func makeMuxRouter() http.Handler {
	muxRouter := mux.NewRouter()
	// muxRouter.HandleFunc("/", handleGetBlockchain).Methods("GET")
	// muxRouter.HandleFunc("/", handleWriteBlock).Methods("POST")
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

// func setupDB() {
// 	// Setup connection to our postgresql database
// 	connString := `user=postgres
// 	password=1234
// 	host=localhost
// 	port=5432
// 	dbname=peopleDatabase
// 	sslmode=disable`

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
