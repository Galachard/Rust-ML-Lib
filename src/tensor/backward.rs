use crate::graph::Node;
use crate::ops::*;
use crate::tensor::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

impl Tensor {
    /// Sort the graph with given root using dfs in reverse topological order (root first)
    fn sort_nodes(root: &Rc<RefCell<Node>>) -> Vec<Rc<RefCell<Node>>> {
        let mut ordered = vec![];
        let mut visited = HashSet::new();

        fn dfs(
            node: &Rc<RefCell<Node>>,
            visited: &mut HashSet<*const Node>,
            ordered: &mut Vec<Rc<RefCell<Node>>>,
        ) {
            let ptr: *const Node = node.as_ptr();

            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            ordered.push(node.clone());
            for parent in &node.borrow().parents {
                dfs(parent, visited, ordered);
            }
        }

        dfs(root, &mut visited, &mut ordered);
        ordered
    }

    /// Perform backward pass to compute gradients
    pub fn backward(&self) {
        // Root node
        let node = self.node.as_ref().expect("No node found").clone();
        node.borrow_mut().grad = Some(Tensor::ones_like(self));
        let order = Self::sort_nodes(&node);

        // Traverse nodes
        for n in order {
            // In this order all grads are guaranteed to be Some() - they will have a value
            // from their children
            let incoming_grad = {
                let nb = n.borrow();
                nb.grad.as_ref().expect("missing grad").clone()
            };

            // Compute gradients for parents using GradFn
            let parent_grads = n.borrow().grad_fn.backward(&incoming_grad);

            // Accumulate gradients into the parents’ grad fields
            for (parent, g) in n.borrow().parents.iter().zip(parent_grads) {
                let mut pb = parent.borrow_mut();

                pb.grad = match &pb.grad {
                    Some(prev) => Some(raw_add(prev, &g).expect("Shape mismatch in backward")),
                    None => Some(g),
                };
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::grad::LeafGrad;

    #[test]
    fn test_dfs_simple_graph() {
        // Create leaf nodes
        let leaf1 = Node::new(vec![], Box::new(LeafGrad));
        let leaf2 = Node::new(vec![], Box::new(LeafGrad));

        // Create an intermediate node that depends on leaf1 and leaf2
        let intermediate = Node::new(vec![leaf1.clone(), leaf2.clone()], Box::new(LeafGrad));

        // Create a root node that depends on the intermediate node
        let root = Node::new(vec![intermediate.clone()], Box::new(LeafGrad));

        // Perform sort
        let order = Tensor::sort_nodes(&root);

        // Ensure that leaf nodes appear before intermediate, which appears before root
        fn after(
            node1: &Rc<RefCell<Node>>,
            node2: &Rc<RefCell<Node>>,
            order: Vec<Rc<RefCell<Node>>>,
        ) -> bool {
            order.iter().position(|n| Rc::ptr_eq(n, node1)).unwrap()
                > order.iter().position(|n| Rc::ptr_eq(n, node2)).unwrap()
        }
        assert!(after(&leaf1, &intermediate, order.clone()));
        assert!(after(&leaf2, &intermediate, order.clone()));
        assert!(after(&intermediate, &root, order.clone()));
    }
}
