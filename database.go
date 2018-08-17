package main

import "database/sql"

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
