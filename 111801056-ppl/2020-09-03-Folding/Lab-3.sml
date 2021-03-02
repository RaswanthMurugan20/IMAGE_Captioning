fun foldr sfun s [] = s
  | foldr sfun s (x::xs) = sfun(x , foldr sfun s xs)

fun foldl sfun s [] = s
  | foldl sfun s (x::xs) = foldl sfun (sfun(x,s)) xs

fun sum [] = 0
  | sum xs = foldl op+ 0 xs
		 
fun reverse [] = []
  | reverse xs = foldl op:: [] xs

fun map f xs =
    let
	fun funmap f(x,y) = f(x)::y
    in
       foldr (funmap f) [] xs	
    end
			      
fun partition f xs =
    let
	fun split f(x,(xs,ys)) =
        if f x
        then (x::xs,ys)
        else(xs,x::ys)
    in
	foldl (split f) ([],[]) xs
    end	
    
datatype 'a option = None | LookingFor of int | Found of 'a
							     		    					       
fun move (x,(LookingFor n)) =
	    if n = 0 then Found x
	    else LookingFor (n-1)
            | move (_ ,(Found x)) = Found x

fun nth ([],n) = None
  | nth (xs,n) = foldl move (LookingFor n) xs 
					     
