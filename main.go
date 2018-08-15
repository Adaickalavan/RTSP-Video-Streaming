package main

import (
	"database/sql"
	"encoding/json"
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

	log.Fatal(run())
	fmt.Println("Program completed")
}

func makeMuxRouter() http.Handler {
	muxRouter := mux.NewRouter()
	muxRouter.HandleFunc("/definition", getDefinition).Methods("GET")
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
