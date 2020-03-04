In Problem 1, firstly read the txt file and make it into DataFrame separated by’,’ whose
columns are ‘user’, ‘Rating’ and ‘data’. Then use movie_row to store the index of items
containing ‘:’ in the first column, which is the movieID. Check the format of the first
and third column to examine whether none are missing in MovieIDs and UserIDs. Use
df.loc to find and keep rates that are equal to or higher than three. Use sum function to
count users who rated more than or equal to 20, use lambda function to remove these
users for further analysis.

Use rows_no_movie to store the data without movie ID and users rated more than 20
movies, and use final_rows_no_movie to store the data that excludes rating less than
three. Use zip to save the data into a dictionary, key is movie, value is list of users for
this movie. After finishing this, program will print "convert into dict completed", and
the users name should be sorted in the dictionary. Then the shape consisting movie
number and user number will be printed out. Then we will use matrix_output to store
the required format in question one, containing all movies and users that have rated
between 1 and 20 at three or above, as following:
