(* Cantor's diag *) 
fun change x = if x = "a" then "z" else "a"
					    
fun ith (x::xs) i = if i = 0 then x
  else if i < 0 then ""
  else ith xs (i-1)
  | ith [] i = "z"

fun lidx ((x:string)::xs) i =
    let
	fun ith (x::xs) i = if i = 0 then x
			    else if i < 0 then ""
			    else ith xs (i-1)
	  | ith [] i = "z"
    in
	if size((ith (x::xs) i)) >= (i+1) then change (substring((ith (x::xs) i),i,1))
	else "z"
    end
  | lidx [] i = "z"


fun len (x::xs) = 1 + (len xs)
  | len [] = 0 		       
		       
fun fresh_ ((x:string)::xs) j i = if (i - j) = 1 then [(lidx (x::xs) (i-1))]
				  else [(lidx (x::xs) j)] @ (fresh_ (x::xs) (j+1) i)
				      
			      
  | fresh_ [] _ _ = [(lidx [] 0)]

val list = ["a","aa","aaa","aaaa","xqxqa","qpokzq","qwecdcz","pokoommx","zzzzzaaaa"]   
val fresh = fresh_ list 0 (len list)
		   
				  


	


	    
					
		 
  
	
		   
