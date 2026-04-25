import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import TrailScreen from './screens/TrailScreen';
import SandboxScreen from './screens/SandboxScreen';
import ExerciseScreen from './screens/ExerciseScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        initialRouteName="Trail"
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: '#131F24' }
        }}
      >
        <Stack.Screen name="Trail" component={TrailScreen} />
        <Stack.Screen name="Exercise" component={ExerciseScreen} />
        <Stack.Screen name="Sandbox" component={SandboxScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
