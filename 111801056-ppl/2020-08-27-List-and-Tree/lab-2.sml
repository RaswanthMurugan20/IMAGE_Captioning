(* map : ('a -> 'b) -> 'a list -> 'b list*)		   
fun map f [] = []
  | map f (x::xs) = f x :: map f xs

(*tree datatype*)
datatype 'a tree = empty |  node of 'a * 'a tree * 'a tree
	       
(*treemap ('a -> 'b) -> 'a tree -> 'b tree *)
fun treemap f(empty) = empty 
  | treemap f(node(a,al,ar)) = node(f(a), treemap f(al), treemap f(ar))

(*inorder traveral 'a tree -> 'a list *)
fun inorder(empty) = []
  | inorder(node(a,al,ar)) = ((inorder(al) @ [a]) @ inorder(ar))

(*preorder traversal 'a tree -> 'a list *)
fun preorder(empty) = []
  | preorder(node(a,al,ar)) = (([a] @ preorder(al)) @ preorder(ar))

(*postorder traversal 'a tree -> 'a list *)
fun postorder(empty) = []
  | postorder(node(a,al,ar)) = ((postorder(al)@postorder(ar)) @ [a]) 

(*clockwise rotate 'a tree -> 'a tree *)
fun crotate(empty) = empty
  | crotate(node(a,node(b,bl,br),ar)) = node(b,bl,node(a,br,ar))  
  | crotate (node(a,empty,ar)) = node(a,empty,ar)  
			       
			   
			       
