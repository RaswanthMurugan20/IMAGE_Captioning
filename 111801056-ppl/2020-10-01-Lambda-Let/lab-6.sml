(*

Q1 - Define abstract syntax for ¦Ë-let and ¦Ë-letrec as a SML datatype.

*)

type var = string
	       
datatype expr =  Var of var
	       | Compose of expr*expr
	       | Lambda of var*expr

datatype letexpr = Let of var*letexpr*letexpr
		 | Lambdalet of var*letexpr
		 | Composelet of letexpr*letexpr
		 | Varlet of expr

datatype letrecexpr = Letrec of var*letrecexpr*letrecexpr
		    | Lambdarec of var*letrecexpr
		    | Composerec of letrecexpr*letrecexpr
		    | Varrec of expr
				    

(*

Q2 - Write the conversion from these language to that of the plain lambda calculus as described in the class. Write the conversion process using two functions unletrec : ¦Ë-letrec -> ¦Ë-let 
and unlet : ¦Ë-let -> ¦Ë-calculus

*)

fun unletrec letrecexp =
    case letrecexp of
	Varrec (x) => Varlet (x)
      | Letrec(x, expr1, expr2) => Let(x, unletrec expr1, unletrec expr2)
      | Composerec(expr1, expr2) => Composelet(unletrec expr1, unletrec expr2)
      | Lambdarec(x, expr1) => Lambdalet(x, unletrec expr1)
					
  
fun unlet letexp =
    case letexp of
	Varlet (x) => x
      | Let(x, expr1, expr2) => Compose(Lambda(x, unlet expr2), unlet expr1)
      | Composelet(expr1, expr2) => Compose(unlet expr1, unlet expr2)
      | Lambdalet(x, expr1) => Lambda(x, unlet expr1)
				     
    

