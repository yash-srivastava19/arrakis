# Read Write Heads

## Introduction

ReadWriteHeads is a tool designed for identifying and analyzing the read and write heads within transformer models. It focuses on distinguishing between heads that predominantly "write" information to the context and those that "read" from it, facilitating a deeper understanding of the model's information processing dynamics.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.read_write_heads
   :members:
   :undoc-members:
   :show-inheritance:
```


- `identify_write_heads(layer_idx, threshold=0.8)`: Identifies the heads within a specified layer that act primarily as write heads, based on a threshold for their contribution to the overall context.

- `identify_read_heads(layer_idx, dim_idx, threshold=0.8)`: Identifies the heads within a specified layer that act primarily as read heads, focusing on their ability to extract information from the context.

- `trace_information_flow(input_ids, src_token, dst_token, n_layers=2)`: Traces the flow of information from a source token to a destination token across a specified number of layers, highlighting the roles of read and write heads in this process.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage ReadWriteHeads for their specific needs.*

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An essential resource for those interested in the mathematical and theoretical foundations of transformer models. It provides a comprehensive exploration of the roles of read and write heads in information processing, making it an invaluable resource for researchers and practitioners looking to deepen their understanding of transformer model dynamics.