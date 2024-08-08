import random

class MockInventoryService:
    def __init__(self):
        self.products = self._generate_products(100)

    def _generate_products(self, count):
        product_names = [
            'Foundation', 'Lipstick', 'Mascara', 'Eyeliner', 'Blush', 'Concealer',
            'Eyeshadow', 'Lip Gloss', 'Bronzer', 'Highlighter', 'Primer', 'Setting Spray'
        ]
        brands = [
            'BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF',
            'BrandG', 'BrandH', 'BrandI', 'BrandJ', 'BrandK', 'BrandL'
        ]
        products = []
        for _ in range(count):
            product = {
                'id': random.randint(1000, 9999),
                'name': random.choice(product_names),
                'brand': random.choice(brands),
                'price': round(random.uniform(5.0, 50.0), 2),
                'stock': random.randint(1, 100)
            }
            products.append(product)
        return products

    def get_all_products(self):
        return self.products

# Usage example
if __name__ == "__main__":
    inventory_service = MockInventoryService()
    products = inventory_service.get_all_products()
    for product in products:
        print(product)
