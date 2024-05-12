refiner_template = """
### Instruction:
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

### Constraints:
- Review Query: Double-check your SQL query to ensure that the column name is spelled correctly.
- Check Table Schema: Verify the schema of the table you're querying against to confirm the exact column names.
- Use Aliases: If using aliases in your query, ensure that you're referencing the correct alias for the column name.
- In SELECT <column>, just select needed columns in the 【Question】 without any unnecessary column or value
- In FROM <table> or JOIN <table>, do not include unnecessary table
- If use max or min func, JOIN <table> FIRST, THEN use SELECT MAX(<column>) or SELECT MIN(<column>)
- If [Value examples] of <column> has 'None' or None, use JOIN <table> or WHERE <column> is NOT NULL is better
- If use ORDER BY <column> ASC|DESC, add GROUP BY <column> before to select distinct values

Here is a typical example:

### Query:
 What is the name and population of the cities with Italian language?
### Database info:

 Table 'city' has columns: ID (INTEGER), Name (CHAR(35)), CountryCode (CHAR(3)), District (CHAR(20)), Population (INTEGER), and foreign keys: ['CountryCode'] -> country.['Code']. The table description is: Stores information about cities. It has attributes like name (Name), country code (CountryCode), district (District), and population (Population). CountryCode is a foreign key referencing the Code field in the country table, indicating the country the city belongs to.
Here are some relevant example rows (values in the same order as columns above)
(1509, 'Catanzaro', 'ITA', 'Calabria', 96700)
(1514, 'Brindisi', 'ITA', 'Apulia', 93454)


Table 'country' has columns: Code (CHAR(3)), Name (CHAR(52)), Continent (TEXT), Region (CHAR(26)), SurfaceArea (DECIMAL(10, 2)), IndepYear (SMALLINT), Population (INTEGER), LifeExpectancy (DECIMAL(3, 1)), GNP (DECIMAL(10, 2)), GNPOld (DECIMAL(10, 2)), LocalName (CHAR(45)), GovernmentForm (CHAR(45)), HeadOfState (CHAR(60)), Capital (INTEGER), Code2 (CHAR(2)), and foreign keys: . The table description is: Stores information about countries. Each country has a unique three-letter code (Code) as the primary key. It holds details like country name (Name), continent (Continent), region (Region), land area (SurfaceArea), independence year (IndepYear) if applicable, population (Population), life expectancy (LifeExpectancy), and various economic data (GNP, GNPOld). It also stores information on government form (GovernmentForm), head of state (HeadOfState), and capital city reference (Capital - likely referencing the ID from the city table). Additional details include local name (LocalName), a secondary country code (Code2), and official language information (linked to the countrylanguage table).
Here are some relevant example rows (values in the same order as columns above)
('ITA', 'Italy', 'Europe', 'Southern Europe', 301316, 1861, 57680000, 79, 1161755, 1145372, 'Italia', 'Republic', 'Carlo Azeglio Ciampi', 1464, 'IT')
('SMR', 'San Marino', 'Europe', 'Southern Europe', 61, 885, 27000, 81.1, 510, None, 'San Marino', 'Republic', None, 3171, 'SM')


Table 'countrylanguage' has columns: CountryCode (CHAR(3)), Language (CHAR(30)), IsOfficial (TEXT), Percentage (DECIMAL(4, 1)), and foreign keys: ['CountryCode'] -> country.['Code']. The table description is: Connects countries with their spoken languages. A combination of CountryCode (referencing the country table) and Language (language name) forms the primary key, ensuring a unique relationship between a country and its languages. It has an IsOfficial flag indicating if the language is officially recognized in the country. It also stores the percentage (Percentage) of the population speaking that language.
Here are some relevant example rows (values in the same order as columns above)
('ITA', 'Italian', 'T', 94.1)
('CAN', 'Italian','F', 1.7)

### old SQL:
 SELECT Name, Population FROM city WHERE Language = 'Italian'
### SQLite error: 
 no such column: Language
### Exception class:
 <class 'sqlite3.OperationalError'>
### correct SQL:
  SELECT Name, Population FROM city WHERE CountryCode = 'ITA'

Now please fixup old SQL and generate new SQL again.

### Query:
-- {query}
### Database info:
{desc_str}
### old SQL:
{sql}
### SQLite error: 
{sqlite_error}
### Exception class:
{exception_class}
### correct SQL:
"""